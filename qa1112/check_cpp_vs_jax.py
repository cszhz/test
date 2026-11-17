"""Rebuild pjrt_run_stablehlo's inputs bit-exactly (glibc srand(0)/rand via ctypes, same float math, bf16 RNE),
run the same model with JAX on TPU, and print mean|y| for comparison with the C++ output."""
import ctypes, importlib.util, pathlib, sys
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, ".")
libc = ctypes.CDLL("libc.so.6")
libc.srand(0)
RAND_MAX = np.float32(2147483647)

def rnd(n, scale):
    raw = np.fromiter((libc.rand() for _ in range(n)), dtype=np.float64, count=n).astype(np.float32)
    f = (raw / RAND_MAX * np.float32(2) - np.float32(1)) * np.float32(scale)
    return jnp.asarray(f).astype(jnp.bfloat16)          # float32 -> bf16, round-to-nearest-even like ToBf16

T, H, O = 4096, 512, 1024
x, r, w = rnd(T * H, 1.0).reshape(T, H), rnd(T * H, 1.0).reshape(T, H), rnd(H, 1.0)
wp = rnd(H * O, np.float32(1) / np.sqrt(np.float32(H))).reshape(H, O)
spec = importlib.util.spec_from_file_location("aot", "aot_stablehlo_export.py")
aot = importlib.util.module_from_spec(spec); spec.loader.exec_module(aot)
y = jax.jit(aot.model)(x, r, w, wp)
print("JAX on TPU: out", y.dtype, y.shape, "mean|y| = %.4f" % float(jnp.mean(jnp.abs(y.astype(jnp.float32)))))
