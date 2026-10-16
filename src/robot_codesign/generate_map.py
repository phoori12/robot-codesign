import mujoco
import re
import mujoco.viewer
import numpy as np
from pathlib import Path

spec = mujoco.MjSpec()
spec.modelname = "field_diagonal_krails"

MESHES_ROOT = (Path(__file__).parent / "../../meshes").resolve()
XML_DIR     = (Path(__file__).parent / "../../xml").resolve()

def rgb255(r, g, b, a=1.0):
    return [r / 255, g / 255, b / 255, a]

def clean_name(stem: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", stem)   # anything else becomes "_"
    name = re.sub(r"_+", "_", name).strip("_")   # merge repeated "_", trim the ends
    if not name or name[0].isdigit():            # don't start with a digit
        name = "part_" + name
    return name.lower()

def generate_map(mesh_dir: Path, out_xml: Path) -> None:
    if "field" not in str(mesh_dir.name): 
        return
    map_name = clean_name(mesh_dir.name)

    spec = mujoco.MjSpec()                 # fresh spec per map; otherwise meshes pile up across maps
    spec.modelname = map_name

    stl_files = sorted(p for p in mesh_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".stl")
    if not stl_files:
        print(f"skip {mesh_dir.name}: no STL files")
        return
    
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
    # for f, n in zip(stl_files, names):
    #     print(f"{f.name:50s} -> {n}")
    OUT_XML = Path(__file__).parent / "../../xml/field_diagonal_krails_gen.xml"

    rng = np.random.default_rng(seed=0)          # fixed seed -> same colours every run

    for f, name in zip(stl_files, names):
        spec.add_mesh(name=name, file=str(f))
        if "floors" in name or "railings" in name or "thin_osb" in name:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                rgba=rgb255(216, 181, 137, 1.0),  # light brown osb 
            )
        elif "lumber" in name or "pitch" in name:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                rgba=rgb255(152, 107, 65, 1.0),   # medium brown lumber
            )
        elif "pvc" in name:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                rgba=rgb255(255, 255, 255, 1.0),   # medium brown lumber
            )
        elif "half_step" in name:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                rgba=rgb255(255, 255, 0, 1.0),   # medium brown lumber
            )
        elif "full_step" in name:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                rgba=rgb255(255, 150, 0, 1.0),   # medium brown lumber
            )
        else:
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
    xml = spec.to_xml()
    xml = re.sub(r"(<mujoco[^>]*>)",
                r'\1\n  <include file="robot_tracked.xml"/>',
                xml, count=1)               # insert right after the opening <mujoco ...> tag
    out_xml.write_text(xml)
    print(f"wrote {out_xml.name} ({len(stl_files)} meshes)")

if __name__ == "__main__":
    for mesh_dir in sorted(d for d in MESHES_ROOT.iterdir() if d.is_dir()):
        out_xml = XML_DIR / f"{clean_name(mesh_dir.name)}.xml"
        try:
            generate_map(mesh_dir, out_xml)
            mujoco.MjModel.from_xml_path(str(out_xml))     # check: map + included robot compile
        except Exception as e:
            print(f"FAILED {mesh_dir.name}: {e}")           # one broken map doesn't stop the rest

