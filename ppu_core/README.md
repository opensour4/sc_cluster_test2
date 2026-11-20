# 模型和训练算子

每个输入包含一组细胞的对照计数、带掩码的响应计数和目标标识。每个细胞的序列以 5 个条件 token 开头，基因按固定顺序置换并切成每组 32 个的 patch。模型分别投影对照和响应，再拼接两者的通道。Transformer 交替沿序列轴和细胞轴做注意力，输出头为每个被掩码的位置计算 48 档计数的 logits。

`layers.py` 使用 PyTorch 算子实现 RMSNorm、split-half RoPE、SwiGLU 和 SDPA。`diffusion.py` 为同组细胞采样共享时间，通过 SciPy 的 Lambert W 函数变换时间分布，再为每个细胞独立生成掩码。`loss.py` 按加噪概率的倒数加权交叉熵。`ema.py` 按已处理的样本数更新幂函数指数移动平均权重。
