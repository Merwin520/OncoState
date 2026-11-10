"""Hard cache and free-space checks performed before every download."""
import shutil
from pathlib import Path


class CacheLimitExceeded(RuntimeError):
    pass


def used_bytes(path):
    root = Path(path)
    return sum(item.stat().st_size for item in root.rglob('*') if item.is_file()) if root.exists() else 0


def check(path, incoming_bytes, max_cache_gb):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    current = used_bytes(root)
    cap = int(max_cache_gb * 10**9)
    if current + int(incoming_bytes) > cap:
        raise CacheLimitExceeded(f'cache guard: {current}+{incoming_bytes}>{cap} bytes')
    free = shutil.disk_usage(root).free
    if int(incoming_bytes) > free:
        raise CacheLimitExceeded(f'free-space guard: {incoming_bytes}>{free} bytes')
    return dict(cache_bytes_before=current, incoming_bytes=int(incoming_bytes), cap_bytes=cap, free_bytes=free)
