# sc_cluster_test2

在目标加速卡上检验单细胞计数模型的训练、推理与评测。默认入口直接读取仓库里的合成数据，使用 **18,533 个基因、每组 128 个细胞、123,543,040 个可训练参数**。测试通过缩短训练和采样步数控制用时。安装依赖后整个流程可离线运行。

## 环境与入口

参考环境为 Linux、Python 3.12、PyTorch 2.7.1、NumPy 2.2.6、SciPy 1.17.1。在 PPU 服务器上先使用厂商配套的驱动、PyTorch 和设备／通信插件，再安装数值依赖。CPU 环境可另用 `requirements-cpu.txt`；CPU 路径同样执行完整模型，需要足够的内存和运行时间。离线安装可向 pip 传入 `--no-index --find-links /path/to/wheels`。

```bash
git clone https://github.com/opensour4/sc_cluster_test2.git
cd sc_cluster_test2
python -m pip install -r requirements.txt
```

由厂商确认设备注册名，并设置 `PPU_DEVICE`。厂商提供 CUDA 兼容接口时可用 `cuda`，需要显式导入后端模块时再设置 `PLUGIN`。

```bash
# 单卡，默认 BF16，3 次更新，每卡每次处理 1 组，累积 16 个微批，随后做 4 步反向采样
bash smoke.sh "${PPU_DEVICE:?请先设置厂商 PyTorch 的设备名}"

# 八卡，PPU_BACKEND 取厂商注册的集合通信后端；设备名填写类型，省略卡号
NPROC=8 BACKEND="${PPU_BACKEND:?请设置通信后端}" bash smoke.sh "$PPU_DEVICE"

# 在 NVIDIA 上复现完整 128 步推理案例
NPROC=8 BACKEND=nccl TRAIN_STEPS=1 INFER_STEPS=128 bash smoke.sh cuda
```

默认每卡 1 组，每组固定 128 个细胞。用 `MICRO_GROUPS=1` 或 `MICRO_GROUPS=2` 选择每个 rank 在同一次前向／反向中处理的组数。双组输入为 256 个细胞，两个组可有不同的目标，训练时逐微批检查真实张量行数和组 ID。单组默认累积 16 次，双组默认累积 8 次；两者每次更新均为每卡 16 组／2,048 个细胞。八卡为 128 组／16,384 个细胞，16 卡在同样设置下为 256 组／32,768 个细胞。`ACCUM` 可显式覆盖累积次数。

在单张 NVIDIA L20Z 上，可用容量为 79.19 GiB。常规 BF16 前向处理 2 组时实测显存不足，释放临时嵌入和改用可扩展分配器后仍在注意力阶段失败。其他设备可以用 `MICRO_GROUPS=2 bash smoke.sh "$PPU_DEVICE"` 测量自身容量；成功时日志应显示每个 rank 为 2 组、256 个细胞。

每次运行都会打印一个新的 `runs/` 子目录。训练使用 AdamW、梯度裁剪、双 EMA 和 1,000 步线性 warmup；多卡由 torchrun 启动，每个 rank 绑定一张卡。推理进程在一张卡上重新加载 EMA 0.05，从全 MASK 的 Y 开始采样，评测在 CPU 上执行。用 `PRECISION=fp32` 单独排查 BF16 问题，用 `RUN_DIR` 指定一个新的输出目录。

## 怎样判定结果正确

`smoke.sh` 顺序执行以下检查。**整条命令退出 0，且最后的数值校验输出 `"pass": true`，表示该环境通过这套迁移测试。** 失败时查看相应阶段的日志和进程退出码。

| 文件 | 实际检查 |
| --- | --- |
| `train.log` | 核对各 rank 的设备和模型参数量；检查每一步各 rank 的梯度范数有限且大于零，并核对 loss、优化器更新及多卡输出头同步 |
| `infer.log`、`eval.log`、`metrics.json` | 核对 EMA 来源、设备精度、计数范围、目标顺序和输出形状。拒绝全零或恒定的目标输出、整块重复的目标预测，以及重复写入的完整细胞向量。用匿名合成数据的解析均值计算 NRE、PDS，并给出逐目标指标 |
| `check_run.log` | 核对本次 checkpoint、预测、数据和评测文件的对应关系，并从预测重新计算指标。命中同配置 NVIDIA 记录时，另外列出末步训练 loss、留出 loss 和 macro NRE 的差值 |
| `numeric.log` | 使用固定初始化、固定输入和掩码，检查 384 个 logits、加权交叉熵、192 个输出头梯度元素和输出头梯度范数。另用 4 组固定输入核对推理所用的 FP64 Gumbel 采样和最终计数 token。数值超出容差或 token 不匹配时返回非零 |

这两类对照回答不同问题。端到端训练和 Gumbel 采样使用设备随机流，跨设备的 loss 与 NRE 允许变化。固定数值校验用相同 CPU 初始化的完整模型、同一组 128 个留出细胞、4,096 个固定掩码位置和独立的 NVIDIA 基准测量。它在目标设备上执行完整前向及输出头反向，按 `绝对误差 ≤ atol + rtol × |参考值|` 逐元素判定。FP32 和 BF16 分别使用自己的参考值和工程容差。完整训练阶段另行执行所有可训练参数的反向传播。[端到端参考值](references/runs.json) · [固定输入数值基准](references/numeric.json) · [阈值与实现](verify.py)

以下命令分别运行合成预测的正反例、重查已有运行产物和校验目标设备的固定输入。

```bash
python -m unittest discover -s tests -v
python check_run.py runs/你的运行目录
python verify.py --device "$PPU_DEVICE" --precision bf16
```

### 已实测的 NVIDIA 对照

以下为 NVIDIA L20Z、CUDA 12.6、PyTorch 2.7.1 的 BF16 实测。下表使用同一份随仓数据，预测形状均为 **2 个目标 × 128 个留出细胞 × 18,533 个基因**。

| 训练 | 每卡 micro batch（组） | 累积次数 | 优化器更新 | 推理步数 | 末步训练 loss | 留出 loss | macro NRE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 单卡 | 1 | 16 | 3 | 4 | 3.949729 | 3.943611 | 1194.370875 |
| 八卡 NCCL | 1 | 16 | 3 | 4 | 3.948084 | 3.943583 | 1194.340469 |
| 八卡 NCCL | 1 | 16 | 1 | 128 | 3.950916 | 3.946332 | 1194.162450 |

短训且随机初始化的模型可以产生很大的 NRE。表中数值用于同配置诊断，不代表预测质量目标。固定案例在 NVIDIA 上的 FP32／BF16 参考损失分别为 3.949609 和 3.949691。

## 模型与数据

模型保留 prefix 5、patch 32、固定基因置换、48 档 count head、13 层交错 L/N 轴 SDPA、宽度 640、10 个注意力头和宽度 2,560 的前馈层。训练和独立推理都会检查上述固定规格。[模型配置](model_config.py)

`data/dummy.npz` 大小为 8,795,316 字节，直接保存 18,533 个基因上的完整计数。训练矩阵包含两个匿名扰动和一个非靶向条件；训练与留出计数分别采样。文件中另有控制及扰动的解析均值，供评测使用。可用 `python fixture.py --output /tmp/dummy-copy.npz` 重新生成；和随仓文件核对时应逐数组比较，跨平台压缩器可能生成不同的文件指纹。固定数值验收使用随仓原文件。[合成数据生成器](fixture.py)

各阶段也可独立执行。保留训练日志，方便事后核对各 rank 实际收到的微批：

```bash
set -o pipefail
mkdir -p runs/manual
python train.py --device cuda --precision bf16 --output runs/manual/checkpoint.pt | tee runs/manual/train.log
python infer.py --device cuda --precision bf16 --checkpoint runs/manual/checkpoint.pt --output runs/manual/predictions.npz
python evaluate.py --predictions runs/manual/predictions.npz --output runs/manual/metrics.json
python check_run.py runs/manual
```

## 迁移覆盖范围

这个包保留训练数学、模型尺寸和主要设备算子。加噪包含 CPU SciPy Lambert W 计算及结果回传，反向采样使用设备 FP64 随机数、对数和排序。RoPE、RMSNorm 及 SwiGLU 使用可移植 PyTorch 实现。原 CUDA 融合扩展、私有在线数据生成与异步预取、双节点通信和生产性能需单独验证。PPU 的结论以目标服务器的实测为准。[算子提取说明](ppu_core/README.md)

项目许可见 [LICENSE](LICENSE)，算子的第三方声明见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。
