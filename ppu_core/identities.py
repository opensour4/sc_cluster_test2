"""Stable identities for the count-token contract and sample streams."""
import hashlib
import json


def canonical(value):
    """Return stable JSON bytes for serializable identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def fingerprint(value):
    """Hash a serializable configuration or identity."""
    return hashlib.sha256(canonical(value)).hexdigest()

def public_seed(domain, *keys):
    """Derive a signed-int64-compatible seed from public task identities."""
    return int.from_bytes(hashlib.sha256(canonical([domain, *keys])).digest()[:8], "little") % (2**63 - 1)
