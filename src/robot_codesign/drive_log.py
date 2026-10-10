import time
import numpy as np
import matplotlib
matplotlib.use("Agg")            # macOS + mjpython: no plot windows from a background thread -> save files
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("stairs.xml")
d = mujoco.MjData(m)
m.flg_surfacevel = True          # needed: all surfacevel are 0 at compile time

R = m.geom_size[m.geom("left_drive_pulley_geom").id][0]                        # drive pulley radius [m]
B = m.body("left_drive_pulley").pos[1] - m.body("right_drive_pulley").pos[1]   # track separation [m]
MAX_SPD = 10                     # motor speed limit [rad/s], same as ctrlrange

R_MAIN = 0.09                    # drive / pivot / rear pulley radius [m]
R_TIP = 0.05                     # flipper tip pulley radius [m]

state = {"v": 0.0, "turn": 0.0, "front": 0.0, "rear": 0.0, "reset": False}   # what the keys change


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
        state["v"] = 0 if state["v"] < 0.0 else 3
    elif key == 263:
        state["turn"] = 0 if state["turn"] < 0 else 1
    elif key == 264:
        state["v"] = 0 if state["v"] > 0.0 else -3
    elif key == 262:
        state["turn"] = 0 if state["turn"] > 0 else -1
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


def apply_controls():
    W_MOVING = 7.0
    W_STILL = 2.0
    turn = state["turn"] * (W_MOVING if state["v"] != 0 else W_STILL)
    v = [(state["v"] - turn * B / 2) / R, (state["v"] + turn * B / 2) / R]

    # desaturation: scale both sides together so the turn ratio is kept
    v_top = max(abs(v[0]), abs(v[1]))
    if v_top > MAX_SPD:
        v[0] = v[0] * MAX_SPD / v_top
        v[1] = v[1] * MAX_SPD / v_top

    # motors
    d.actuator("drive_left").ctrl[0] = v[0]
    d.actuator("drive_right").ctrl[0] = v[1]

    # belts follow the motors' ACTUAL speed (hybrid model)
    for side in ("left", "right"):
        omega = d.qvel[m.joint(f"{side}_drive_joint").dofadr[0]]
        set_belt_speed(side, omega * R_MAIN)

    # flippers
    d.actuator("p_front_left").ctrl[0] = state["front"]
    d.actuator("p_front_right").ctrl[0] = state["front"]
    d.actuator("p_rear_left").ctrl[0] = state["rear"]
    d.actuator("p_rear_right").ctrl[0] = state["rear"]


# ---------------------------------------------------------------- logging ---
log = []
base_id = m.body("base_link").id
dof_left = m.joint("left_drive_joint").dofadr[0]
dof_right = m.joint("right_drive_joint").dofadr[0]


def record():
    R_base = d.xmat[base_id].reshape(3, 3)
    log.append({
        "t_sim": d.time,
        "t_wall": time.time(),
        "yaw": np.arctan2(R_base[1, 0], R_base[0, 0]),       # heading [rad]
        "yaw_rate": d.qvel[5],                              # turn rate of the free joint [rad/s]
        "omega_left": d.qvel[dof_left],
        "omega_right": d.qvel[dof_right],
        "force_left": d.actuator("drive_left").force[0],    # motor torque [N·m]
        "force_right": d.actuator("drive_right").force[0],
        "ncon": d.ncon,
    })


# ------------------------------------------------------------- main loop ---
FRAME = 1 / 60                                    # redraw 60 times per second
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
    while viewer.is_running():
        t0 = time.time()
        if state["reset"]:
            mujoco.mj_resetData(m, d)
            state.update(front=0.0, rear=0.0, v=0.0, turn=0.0, reset=False)
            log.clear()                           # start a fresh log after a reset
        apply_controls()
        for _ in range(int(FRAME / m.opt.timestep)):   # several physics steps per frame
            mujoco.mj_step(m, d)
        record()
        viewer.sync()
        time.sleep(max(0.0, FRAME - (time.time() - t0)))   # keep it at real-time speed

# --------------------------------------------------------------- plotting ---
if len(log) < 2:
    raise SystemExit("nothing logged")

L = {k: np.array([row[k] for row in log]) for k in log[0]}
np.savez("turn_log.npz", **L)    # raw data, so you can re-plot later without re-running the sim
t = L["t_sim"]
limit = m.actuator_forcerange[m.actuator("drive_left").id][1]

fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
ax[0].plot(t, L["yaw_rate"])
ax[0].set_ylabel("turn rate [rad/s]")

ax[1].plot(t, L["force_left"], label="left")
ax[1].plot(t, L["force_right"], label="right")
ax[1].axhline(limit, ls="--", color="gray")
ax[1].axhline(-limit, ls="--", color="gray", label=f"limit ±{limit:.0f}")
ax[1].set_ylabel("motor torque [N·m]")
ax[1].legend(loc="upper right")

ax[2].plot(t, L["ncon"])
ax[2].set_ylabel("contacts")

ax[3].plot(t, L["t_wall"] - L["t_wall"][0], label="wall clock")
ax[3].plot(t, t - t[0], ls="--", label="ideal (real time)")
ax[3].set_ylabel("elapsed wall time [s]")
ax[3].set_xlabel("simulation time [s]")
ax[3].legend(loc="upper left")
fig.suptitle("Turning diagnostics")
fig.tight_layout()
fig.savefig("turn_diagnostics.png", dpi=150)

# turn rate against heading: do the dips happen at the same angle every rotation?
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.scatter(np.degrees(L["yaw"]), L["yaw_rate"], s=4)
ax2.set_xlabel("heading [deg]")
ax2.set_ylabel("turn rate [rad/s]")
ax2.set_title("Turn rate vs heading")
fig2.tight_layout()
fig2.savefig("turn_vs_heading.png", dpi=150)

print("saved turn_diagnostics.png, turn_vs_heading.png and turn_log.npz")