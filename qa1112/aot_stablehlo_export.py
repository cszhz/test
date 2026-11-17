"""JIT / AOT / StableHLO 导出示例（第三部分）。

演示：
1. JIT：jax.jit 第一次调用时编译，之后复用；可开持久编译缓存，进程重启后不用重新编译。
2. AOT 两条路：
   a) jit(f).lower(...) -> Lowered（可拿 StableHLO 文本）-> .compile() -> Compiled（提前编好，运行时不再编译）
   b) jax.export.export(jit(f), platforms=["tpu"])(*shapes) -> Exported
      .serialize() 得到可落盘的字节（StableHLO + 调用约定 + 版本号），另一个进程 / 兼容的 JAX 版本
      jax.export.deserialize(...) 后 .call(...) 直接执行。可以在没有 TPU 的机器上导出 TPU 版本。
3. 带 Pallas 自定义 kernel 的函数也能导出：Pallas kernel 在 StableHLO 里是一个 custom_call
   "tpu_custom_call"（内含序列化的 Mosaic kernel），它在 jax.export 的"保证稳定"白名单里。
4. 写出 StableHLO bytecode（model.stablehlo.mlirbc）和 XLA 编译选项（compile_options.pb），
   给 C++ 通过 PJRT C API 加载 libtpu.so 执行（见 pjrt_run_stablehlo.cc）。

示例模型：y = rmsnorm_pallas(x + r) @ W（RmsNorm 用 rmsnorm.py 里的 Pallas kernel）。

运行
  任何机器（导出 TPU 产物，不需要 TPU）：JAX_PLATFORMS=cpu python aot_stablehlo_export.py --out artifacts
  TPU 上（另外做 lower/compile、反序列化后执行并对比）：python aot_stablehlo_export.py --out artifacts
"""

import argparse
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
from jax import export as jexport

_spec = importlib.util.spec_from_file_location(
    "rmsnorm", pathlib.Path(__file__).with_name("rmsnorm.py"))
rmsnorm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmsnorm)

TOKENS, HIDDEN, OUT = 4096, 512, 1024


def model(x, r, w_norm, w_proj):
    h, _ = rmsnorm.residual_rmsnorm_pallas(x, r, w_norm, block_rows=512)
    return jnp.dot(h, w_proj, preferred_element_type=jnp.float32).astype(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    on_tpu = jax.default_backend() == "tpu"

    shapes = (jax.ShapeDtypeStruct((TOKENS, HIDDEN), jnp.bfloat16),
              jax.ShapeDtypeStruct((TOKENS, HIDDEN), jnp.bfloat16),
              jax.ShapeDtypeStruct((HIDDEN,), jnp.bfloat16),
              jax.ShapeDtypeStruct((HIDDEN, OUT), jnp.bfloat16))
    jitted = jax.jit(model)

    # --- 2b. jax.export：为 TPU 导出（在 CPU 上也能做） ---
    exported = jexport.export(jitted, platforms=["tpu"])(*shapes)
    text = exported.mlir_module()
    (out / "model.stablehlo.mlir").write_text(text)
    (out / "model.stablehlo.mlirbc").write_bytes(exported.mlir_module_serialized)
    blob = exported.serialize()
    (out / "model.jax_exported").write_bytes(bytes(blob))
    print(f"导出平台 {exported.platforms}，调用约定 v{exported.calling_convention_version}")
    print(f"输入 {[str(a) for a in exported.in_avals]} -> 输出 {[str(a) for a in exported.out_avals]}")
    print(f"StableHLO 文本 {len(text) / 1e3:.1f} KB，bytecode {len(exported.mlir_module_serialized) / 1e3:.1f} KB，"
          f"序列化 Exported {len(blob) / 1e3:.1f} KB")
    print(f"含 Pallas kernel（tpu_custom_call）：{'tpu_custom_call' in text}")

    # 反序列化（另一个进程里做法相同）
    restored = jexport.deserialize(bytearray((out / "model.jax_exported").read_bytes()))
    print(f"反序列化后平台 {restored.platforms}，输入数 {len(restored.in_avals)}")

    # --- 4. 给 C++/PJRT 的编译选项 ---
    try:
        from jax._src import compiler
        opts = compiler.get_compile_options(num_replicas=1, num_partitions=1)
        (out / "compile_options.pb").write_bytes(opts.SerializeAsString())
        print("已写 compile_options.pb（xla.CompileOptionsProto）")
    except Exception as exc:  # 私有 API，版本间可能变
        print(f"跳过 compile_options.pb：{exc!r}")

    if not on_tpu:
        print("当前不是 TPU：只做了导出。到 TPU 上再运行本脚本可做执行对比。")
        return

    # --- 1/2a/2b 在 TPU 上执行 ---
    keys = jax.random.split(jax.random.key(0), 4)
    args_ = [jax.random.normal(k, s.shape, s.dtype) for k, s in zip(keys, shapes)]
    y_jit = jitted(*args_)                                  # 1. JIT（首次调用编译）
    compiled = jitted.lower(*shapes).compile()               # 2a. AOT 提前编译
    y_aot = compiled(*args_)
    y_exp = restored.call(*args_)                            # 2b. 反序列化后执行
    for name, y in [("AOT compiled", y_aot), ("deserialized export", y_exp)]:
        err = float(jnp.max(jnp.abs(y.astype(jnp.float32) - y_jit.astype(jnp.float32))))
        print(f"{name} vs jit: max|err| = {err:.3g}")


if __name__ == "__main__":
    main()
