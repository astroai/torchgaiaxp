"""Throughput: torchgaiaxp vs GaiaXPy (CPU here; run on a GPU node for CUDA).

python scripts/benchmark.py 20000 cuda
"""

import sys
import time

import torch

sys.path.insert(0, "tests")
from conftest import frame, make_batch

from torchgaiaxp import XPSpectra

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
batch = make_batch(n, realistic=False)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


for dtype in (torch.float32, torch.float64):
    xp = XPSpectra.from_batch(batch, device=device, dtype=dtype)
    for name, fn in (
        ("calibrate", lambda xp=xp: xp.calibrate()),
        ("calibrate+cov factor", lambda xp=xp: xp.calibrate(with_correlation=True)["factor"]),
        ("generate SDSS_Std", lambda xp=xp: xp.generate("SDSS_Std")),
        (
            "eval 2000 custom wl",
            lambda xp=xp, dtype=dtype: xp(torch.linspace(340, 1000, 2000, device=device, dtype=dtype)),
        ),
    ):
        fn()
        sync()
        start = time.perf_counter()
        fn()
        sync()
        print(
            f"{device} {str(dtype)[6:]} {name}: {n / (time.perf_counter() - start):,.0f} sources/s"
        )

try:
    import gaiaxpy

    m = min(n, 300)
    f = frame(make_batch(m, realistic=False))
    start = time.perf_counter()
    gaiaxpy.calibrate(f, save_file=False)
    print(f"gaiaxpy calibrate: {m / (time.perf_counter() - start):,.0f} sources/s")
    start = time.perf_counter()
    gaiaxpy.generate(f, photometric_system=gaiaxpy.PhotometricSystem.SDSS_Std, save_file=False)
    print(f"gaiaxpy generate SDSS_Std: {m / (time.perf_counter() - start):,.0f} sources/s")
except ImportError:
    pass
