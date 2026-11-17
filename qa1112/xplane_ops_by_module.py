"""按 jit 模块拆分 xplane 里的 XLA op 时间：python xplane_ops_by_module.py <xplane.pb> [plane]"""
import collections, sys
from jax._src.profiler import ProfileData
pb, plane_name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "/device:TPU:0")
plane = next(p for p in ProfileData.from_file(pb).planes if p.name == plane_name)
lines = {l.name: l for l in plane.lines}
mods = [(e.start_ns, e.start_ns + e.duration_ns, e.name.split("(")[0]) for e in lines["XLA Modules"].events]
stat = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
count = collections.Counter(m[2] for m in mods)
for e in lines["XLA Ops"].events:
    for s, t, name in mods:
        if s <= e.start_ns < t:
            st = stat[name][e.name.split(".")[0]]
            st[0] += 1; st[1] += e.duration_ns / 1e6
            break
for name, ops in stat.items():
    n = count[name]
    print(f"== {name}  (per call)")
    for op, (c, ms) in sorted(ops.items(), key=lambda kv: -kv[1][1]):
        print(f"   {op:40s} x{c / n:4.1f}  {ms / n:7.3f} ms")
