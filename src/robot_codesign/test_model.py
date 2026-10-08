import mujoco
import numpy as np
import matplotlib.pyplot as plt

m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)

print("Model Bodies: {}".format(m.nbody))
print("Model Joints: {}".format(m.njnt))
print("Model DOFs: {}".format(m.nv))
print("Model NQ: {}".format(m.nq))
print("Model NU: {}".format(m.nu))

# print("Bodies: ")
# for i in range(m.nbody):
#     print(m.body(i).name)

# print("Joints: ")
# for i in range(m.njnt):
#     print(m.joint(i).name)

print("Model Mass: {}".format(m.body_mass.sum()))

for i in range(10):
    q0 = d.qpos.copy()
    mujoco.mj_step(m, d)
    q = d.qpos - q0
    # if i < 3 or i > 997:
    #     print(d.time)
    #     print(d.qpos)

# mujoco.mj_saveLastXML("robot.xml", m)
print(d.ncon)
print(m.opt.timestep)
print(d.actuator_force)
# for i in range(d.ncon):
#     print(d.contact[i])

JOINT = "front_left_flipper_joint"
ACT = "p_front_left"
TARGET = -0.15   # small step (tip up): kp * 0.15 = 135 N·m stays below the 160 N·m limit,
                 # so the servo acts like an ideal spring and the theory applies


def joint_inertia(m, joint_name):
    """Effective inertia (kg·m²) felt by one joint, from the mass matrix."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)               # your MuJoCo version's signature
    dof = m.joint(joint_name).dofadr[0]
    return M[dof, dof]


def set_linear_damping(m, joint_name, c):
    """Set only the linear damping term for one joint."""
    dof = m.joint(joint_name).dofadr[0]
    if np.ndim(m.dof_damping) == 1:
        m.dof_damping[dof] = c
    else:                                  # in case your version stores [a, b, c] per joint
        m.dof_damping[dof] = [c, 0.0, 0.0]


def step_response(m, settle=0.5, duration=1.0):
    """Let the robot settle, step the flipper target, record the flipper angle."""
    d = mujoco.MjData(m)                   # fresh state each run, so runs are comparable
    qadr = m.joint(JOINT).qposadr[0]
    while d.time < settle:
        mujoco.mj_step(m, d)
    d.actuator(ACT).ctrl[0] = TARGET
    t0, times, angles = d.time, [], []
    while d.time - t0 < duration:
        mujoco.mj_step(m, d)
        times.append(d.time - t0)
        angles.append(d.qpos[qadr])
    return np.array(times), np.array(angles)


m = mujoco.MjModel.from_xml_path("scene.xml")
dof = m.joint(JOINT).dofadr[0]
print("damping stored for this joint:", m.dof_damping[dof])   # check this is 0 after your XML fix

I = joint_inertia(m, JOINT)
kp = m.actuator_gainprm[m.actuator(ACT).id, 0]                 # read kp from the model, no magic numbers
c_crit = 2 * np.sqrt(kp * I)
f_pred = np.sqrt(kp / I) / (2 * np.pi)
print(f"I = {I:.4f} kg·m²   kp = {kp:.0f}   predicted f = {f_pred:.1f} Hz   critical c = {c_crit:.1f}")

cases = {"no damping": 0.0,
         f"critical (c = {c_crit:.1f})": c_crit,
         "c = 400 (old value)": 400.0}

for label, c in cases.items():
    set_linear_damping(m, JOINT, c)
    t, q = step_response(m)
    plt.plot(t, q, label=label)

plt.axhline(TARGET, ls="--", color="gray", label="target")
plt.xlabel("time after step [s]")
plt.ylabel("flipper angle [rad]")
plt.title(f"Flipper step response (predicted undamped f = {f_pred:.1f} Hz)")
plt.legend()
plt.show()
