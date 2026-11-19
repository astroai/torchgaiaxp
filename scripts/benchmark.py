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
    xp_cov = xp[: min(n, 8192)]
    for count, name, fn in (
        (n, "calibrate", lambda xp=xp: xp.calibrate()),
        (len(xp_cov), "calibrate+cov factor", lambda x=xp_cov: x.calibrate(with_correlation="factor")["factor"]),
        (n, "generate DR3_SDSS_Std", lambda xp=xp: xp.generate("DR3_SDSS_Std")),
        (
            n,
            "generate 4 sys + err_corr",
            lambda xp=xp: xp.generate(
                ["DR3_SDSS_Std", "DR3_JKC_Std", "DR3_DECam", "DR3_PanSTARRS1_Std"],
                error_correction=True,
            ),
        ),
        (
            n,
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
            f"{device} {str(dtype)[6:]} {name}: {count / (time.perf_counter() - start):,.0f} sources/s"
        )
    del xp, xp_cov
    if device == "cuda":
        torch.cuda.empty_cache()

try:
    import gaiaxpy

    m = min(n, 300)
    f = frame(make_batch(m, realistic=False))
    start = time.perf_counter()
    gaiaxpy.calibrate(f, save_file=False, disable_info=True)
    print(f"gaiaxpy calibrate: {m / (time.perf_counter() - start):,.0f} sources/s")
    start = time.perf_counter()
    gaiaxpy.generate(
        f,
        photometric_system=gaiaxpy.PhotometricSystem.DR3_SDSS_Std,
        save_file=False,
        disable_info=True,
    )
    print(f"gaiaxpy generate DR3_SDSS_Std: {m / (time.perf_counter() - start):,.0f} sources/s")
except ImportError:
    pass
