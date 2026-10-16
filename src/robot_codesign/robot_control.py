import time
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("xml/field_diagonal_krails.xml")
d = mujoco.MjData(m)
d.qpos[0:3] = [0, 0, 1]          # position
d.qpos[3:7] = [1, 0, 0, 0]       # orientation (w, x, y, z)

m.flg_surfacevel = True

R = m.geom_size[m.geom("left_drive_pulley_geom").id][0]                        # drive pulley radius [m]
B = m.body("left_drive_pulley").pos[1] - m.body("right_drive_pulley").pos[1]   # track separation [m]
MAX_SPD = 10

R_MAIN = 0.09     # drive / pivot / rear pulley radius [m]
R_TIP = 0.05      # flipper tip pulley radius [m]

state = {"v": 0.0, "turn": 0.0, "front": 0, "rear": 0.0, "reset": False}   # what the keys change

def set_belt_speed(side, v):
    """Belt speed v [m/s] on one side (+ = robot moves forward), for main track + both flippers."""
    def sv(name, vals):
        m.geom_surfacevel[m.geom(name).id] = vals

    sv(f"track_{side}_rear_pulley", [0, 0, 0, 0, 0, v / R_MAIN])
    sv(f"track_{side}_bottom", [-v, 0, 0, 0, 0, 0])
    sv(f"track_{side}_top", [v, 0, 0, 0, 0, 0])
    for end in ("front", "rear"):
        sv(f"flip_{end}_{side}_pivot", [0, 0, 0, 0, 0, v / R_MAIN])
        sv(f"flip_{end}_{side}_tip", [0, 0, 0, 0, 0, v / R_TIP])
        sv(f"flip_{end}_{side}_bottom", [-v, 0, 0, 0, 0, 0])
        sv(f"flip_{end}_{side}_top", [v, 0, 0, 0, 0, 0])

def on_key(key):
    if key == 265:
        if state["v"] < 0.0:
            state["v"] = 0
        else:
            state["v"] = 0.9
    elif key == 263:
        if state["turn"] < 0:
            state["turn"] = 0
        else:
            state["turn"] = 1
  
    elif key == 264:
        if state["v"] > 0.0:
            state["v"] = 0
        else:
            state["v"] = -0.9
    elif key == 262:
        if state["turn"] > 0:
            state["turn"] = 0
        else:
            state["turn"] = -1
    elif key == ord("1"):
        state["front"] = np.clip(state["front"] + 0.1, -1.6, 1.6)
    elif key == ord("2"):
        state["front"] = np.clip(state["front"] - 0.1, -1.6, 1.6)
    elif key == ord("3"):
        state["rear"] = np.clip(state["rear"] + 0.1, -1.6, 1.6)
    elif key == ord("4"):   
        state["rear"] = np.clip(state["rear"] - 0.1, -1.6, 1.6)
    elif key == 32:
        state["v"] = 0
        state["turn"] = 0
    elif key == ord("R"):
        state["reset"] = True

    # print(state)
    pass

def apply_controls():
    W_MOVING = 7.0
    W_STILL = 2.0
    turn = state["turn"] * (W_MOVING if state["v"] != 0 else W_STILL)
    v = [(state["v"] - turn * B/2) / R, (state["v"] + turn * B/2)/R]
    v_top = max(abs(v[0]), abs(v[1])) 

    if v_top > MAX_SPD:
        v[0] = v[0] * MAX_SPD / v_top
        v[1] = v[1] * MAX_SPD / v_top

    d.actuator("drive_left").ctrl[0] = v[0]
    d.actuator("drive_right").ctrl[0] = v[1]

    for side in ("left", "right"):
        omega = d.qvel[m.joint(f"{side}_drive_joint").dofadr[0]]
        set_belt_speed(side, omega * R_MAIN)

    d.actuator("p_front_left").ctrl[0] = state["front"]
    d.actuator("p_front_right").ctrl[0] = state["front"]
    d.actuator("p_rear_left").ctrl[0] = state["rear"]
    d.actuator("p_rear_right").ctrl[0] = state["rear"]
    pass

FRAME = 1 / 60                                    # redraw 60 times per second
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
    while viewer.is_running():
        t0 = time.time()
        apply_controls()
        if state["reset"]:
            mujoco.mj_resetData(m, d)
            state["front"] = 0.0
            state["rear"] = 0.0
            state["v"] = 0.0
            state["turn"] = 0
            state["reset"] = False
            d.qpos[0:3] = [0, 0, 1]          # position
            d.qpos[3:7] = [1, 0, 0, 0]       # orientation (w, x, y, z)
        for _ in range(int(FRAME / m.opt.timestep)):   # several physics steps per frame
            mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(max(0.0, FRAME - (time.time() - t0)))   # keep it at real-time speed