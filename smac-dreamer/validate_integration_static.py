from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path.cwd()
WORLD = ROOT / "src" / "smacdreamer" / "jepa" / "world_model.py"
ADAPTER = ROOT / "src" / "smacdreamer" / "jepa" / "feature_adapter.py"
CONFIG = ROOT / "configs" / "r2_2100_jepa_local.yaml"
CKPT = ROOT / "checkpoints" / "jepa" / "model.pt"


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"OK: {msg}")


for p in [WORLD, ADAPTER, CONFIG, CKPT]:
    if not p.exists():
        fail(f"missing {p}")
    ok(f"exists {p}")

world_src = WORLD.read_text()
adapter_src = ADAPTER.read_text()

for needle in ["def _belief_mask", "def _seen_mask_from_memory", "num_entities=self.entities"]:
    if needle not in world_src:
        fail(f"world_model.py missing {needle!r}")
ok("world_model.py contains belief-mask helpers and passes num_entities")

if "Pool per-entity JEPA state" in adapter_src or "pooled =" in adapter_src:
    fail("feature_adapter.py still appears to be the old masked-mean-pooling adapter")
for needle in ["Slot-preserving", "reshape(x.shape[0], self.num_entities * self.hidden_dim)"]:
    if needle not in adapter_src:
        fail(f"feature_adapter.py missing {needle!r}")
ok("feature_adapter.py appears slot-preserving, not mean-pooling")

# AST-ish static check: in get_feat(), feature_adapter should occur after the with torch.no_grad block.
tree = ast.parse(world_src)
cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FrozenJEPAWorldModel"), None)
if cls is None:
    fail("FrozenJEPAWorldModel class not found")
get_feat = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_feat"), None)
if get_feat is None:
    fail("get_feat not found")

feature_lines = []
with_lines = []
for node in ast.walk(get_feat):
    if isinstance(node, ast.With):
        with_lines.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
    if isinstance(node, ast.Attribute) and node.attr == "feature_adapter":
        feature_lines.append(node.lineno)
if not feature_lines:
    fail("self.feature_adapter call not found in get_feat")
for fl in feature_lines:
    for wl0, wl1 in with_lines:
        if wl0 <= fl <= wl1:
            fail(f"feature_adapter is still inside a with-block in get_feat at line {fl}")
ok("feature_adapter is outside the no_grad block in get_feat")

print("PASS: static integration checks passed")
