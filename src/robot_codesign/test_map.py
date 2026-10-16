import mujoco
import numpy as np
import matplotlib.pyplot as plt

m = mujoco.MjModel.from_xml_path("assets/field_diagonal_krails/urdf/field_diagonal_krails.urdf")
d = mujoco.MjData(m)

mujoco.mj_saveLastXML("diagonal_krails.xml", m)


print("Model Bodies: {}".format(m.nbody))
print("Model Joints: {}".format(m.njnt))
print("Model DOFs: {}".format(m.nv))
print("Model NQ: {}".format(m.nq))