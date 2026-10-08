import time
import numpy as np
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("stairs.xml")
d = mujoco.MjData(m)

R = m.geom_size[m.body_geomadr[m.body("left_track_wheel_0").id]][0]       # drive wheel radius [m]
B = m.body("left_track_wheel_0").pos[1] - m.body("right_track_wheel_0").pos[1]
MAX_SPD = 10

state = {"v": 0.0, "turn": 0.0, "front": 0, "rear": 0.0, "reset": False}   # what the keys change

def on_key(key):
    if key == 265:
        if state["v"] < 0.0:
            state["v"] = 0
        else:
            state["v"] = 3
    elif key == 263:
        if state["turn"] < 0:
            state["turn"] = 0
        else:
            state["turn"] = 1
  
    elif key == 264:
        if state["v"] > 0.0:
            state["v"] = 0
        else:
            state["v"] = -3
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
    # TODO: compute left/right wheel commands (rad/s) and flipper targets from `state`
    #       and write them into d.ctrl, e.g. with d.actuator("v_left_0").ctrl[0] = ...
    W_MOVING = 7.0
    W_STILL = 2.0
    turn = state["turn"] * (W_MOVING if state["v"] != 0 else W_STILL)
    v = [(state["v"] - turn * B/2) / R, (state["v"] + turn * B/2)/R]
    v_top = max(abs(v[0]), abs(v[1])) 
    
    if v_top > MAX_SPD:
        v[0] = v[0] * MAX_SPD / v_top
        v[1] = v[1] * MAX_SPD / v_top

    drive_actuators = ["v_left_", "v_right_"]

    j = 0
    for da in drive_actuators:
        for i in range(6):
            d.actuator(da+str(i)).ctrl[0] = v[j]
        j = j + 1

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
        for _ in range(int(FRAME / m.opt.timestep)):   # several physics steps per frame
            mujoco.mj_step(m, d)
        viewer.sync()
        time.sleep(max(0.0, FRAME - (time.time() - t0)))   # keep it at real-time speed