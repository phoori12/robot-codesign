import mujoco

m = mujoco.MjModel.from_xml_path("scene.xml")

# every joint name in the model
names = [m.joint(i).name for i in range(m.njnt)]

drive_side = ["right_track_wheel_", "left_track_wheel_"]
drive_end = ["5_joint", "0_joint"]

wheels = [[],[],[],[]]
# the wheel joints of ONE flipper, found by their name pattern
prefix = "left_front_flipper_wheel_"
wheels[2] = [n for n in names if n.startswith(prefix)]
prefix = "right_front_flipper_wheel_"
wheels[0] = [n for n in names if n.startswith(prefix)]

prefix = "left_rear_flipper_wheel_"
wheels[3] = [n for n in names if n.startswith(prefix)]

prefix = "right_rear_flipper_wheel_"
wheels[1] = [n for n in names if n.startswith(prefix)]

# print(wheels)


def wheel_radius(m, joint_name):
    body_id = m.joint(joint_name).bodyid[0]   # the body this joint moves (the wheel)
    geom_id = m.body_geomadr[body_id]         # that body's first (here: only) geom
    return m.geom_size[geom_id][0]            # for a cylinder, size = [radius, half-length, unused]


def belt_line(name, follower, drive, a1):
    """One equality constraint: follower angle = a1 * drive angle."""
    return (f'    <joint name="{name}" joint1="{follower}" joint2="{drive}" '
            f'polycoef="0 {a1:.4f} 0 0 0"/>')


lines = ["  <equality>"]

i = 0
for ds in drive_side:
    for de in drive_end:
        # print(ds + de, wheel_radius(m, ds+de))
        for n in wheels[i]:
            lines.append(belt_line(f"belt_{n}", n, ds+de, wheel_radius(m, ds+de) / wheel_radius(m,n)))
        i = i + 1

# --- your loop goes here ---------------------------------------------
# for each flipper and each of its wheels:
#     lines.append(belt_line(f"belt_{follower}", follower, drive, a1))
# ----------------------------------------------------------------------

lines.append("  </equality>")
xml = "\n".join(lines)
print(xml)