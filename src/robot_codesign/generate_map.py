import mujoco
import re
import mujoco.viewer
import numpy as np
from pathlib import Path

spec = mujoco.MjSpec()
spec.modelname = "field_diagonal_krails"

# Step 2: collect the STL files
MESH_DIR = (Path(__file__).parent / "../../meshes/diagonal_krails").resolve()

# suffix.lower() catches both .STL and .stl; sorted() keeps the order stable between runs
stl_files = sorted(p for p in MESH_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() == ".stl")

if not stl_files:
    raise FileNotFoundError(f"No STL files found in {MESH_DIR}")

# Step 3: turn file names into valid, unique names
def clean_name(stem: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", stem)   # anything else becomes "_"
    name = re.sub(r"_+", "_", name).strip("_")   # merge repeated "_", trim the ends
    if not name or name[0].isdigit():            # don't start with a digit
        name = "part_" + name
    return name.lower()

names = []
used = set()
for f in stl_files:
    base = clean_name(f.stem)                    # .stem = file name without extension
    name, i = base, 2
    while name in used:                          # on a clash, append _2, _3, ...
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    names.append(name)

for f, n in zip(stl_files, names):
    print(f"{f.name:50s} -> {n}")



# ... steps 2 + 3 from before (stl_files, names) ...

OUT_XML = Path(__file__).parent / "field_diagonal_krails_gen.xml"

# Step 4: one mesh + one geom per STL
rng = np.random.default_rng(seed=0)          # fixed seed -> same colours every run

for f, name in zip(stl_files, names):
    spec.add_mesh(name=name, file=str(f))
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        rgba=[*rng.uniform(0.2, 0.9, size=3), 1.0],   # random colour, opaque
    )

# Step 5: floor + light
spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20, 20, 0.1],                      # half-sizes in x/y; 3rd value = grid spacing for rendering
    rgba=[0.5, 0.5, 0.5, 1],
)
spec.worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,)

# Step 6: compile + sanity checks
model = spec.compile()                       # missing files / duplicate names fail here
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)               # computes geom world positions

print(f"{model.nmesh} meshes loaded (expected {len(stl_files)})")

# World-space bounding box of the whole map
pts = []
for g in range(model.ngeom):
    mid = model.geom_dataid[g]
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or mid < 0:
        continue
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[adr:adr + num]
    R = data.geom_xmat[g].reshape(3, 3)
    pts.append(v @ R.T + data.geom_xpos[g])
pts = np.vstack(pts)
lo, hi = pts.min(axis=0), pts.max(axis=0)
print(f"map bounds  min {lo.round(3)}  max {hi.round(3)}  size {(hi - lo).round(3)} m")

# Step 7: save + view
OUT_XML.write_text(spec.to_xml())
print(f"wrote {OUT_XML}")
mujoco.viewer.launch(model)

