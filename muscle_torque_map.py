import sys
import mujoco as m
import numpy as np

sys.modules['numpy'] = np

import time

import record

IMG_HEIGHT = 1088
IMG_WIDTH = 1088
SIMLEN = 0.3  # seconds
FPS = 30
TIMESTEP = 0.001
TORQUE_STEPS=10
ANGLE_STEPS=10


spec = m.MjSpec.from_file("models/myo_weld/myoelbow_0dof6muscles.xml")

spec.option.timestep = TIMESTEP
spec.option.gravity[2] = 0.0  # No gravity for torque mapping

#TODO: remove other muscles so we don't have elasticty

weld = spec.add_equality()
weld.type = m.mjtEq.mjEQ_WELD
weld.name = "elbow_weld"
weld.name1="forearm"
weld.name2="arm_rotation"
weld.objtype=m.mjtObj.mjOBJ_SITE
model = spec.compile()

elbow_axis=spec.joint("r_elbow_flex").axis
rotation=20. #degrees
spec.site("arm_rotation").delete()  # type: ignore
spec.body("r_ulna_radius_hand").add_site(name="arm_rotation", pos=[0, 0, 0], axisangle=np.append(elbow_axis, -np.deg2rad(rotation)))

for tendon in spec.tendons:
    tendon : m.MjsTendon
    tendon.stiffness = 0.0  # Remove tendon elasticity for pure torque mapping
    tendon.damping = 0.0  # Remove tendon damping for pure torque mapping
    tendon.frictionloss = 0.0
    tendon.limited=False

for muscle in spec.actuators:
    muscle : m.MjsActuator
    if muscle.name != "BRA":
        muscle.delete()  # type: ignore

model : m.MjModel= spec.compile()
data : m.MjData= m.MjData(model)



m.mj_forward(model, data)

simstart = data.time




muscle = model.actuator("BRA")

elbow_angle_series = []
torque_series = []
muscle_activation_series = []

# sim loop
while (data.time - simstart) < SIMLEN:

    # apply ctrl
    data.ctrl[muscle.id] = 0.1


    elbow_angle_series.append((data.time, np.rad2deg(data.qpos[0].copy())))
    muscle_activation_series.append((data.time, data.ctrl[muscle.id].copy()))

    torque_series.append((data.time, data.qfrc_actuator[0].copy()))


    # Step the simulation
    m.mj_step(model, data)

final_torque=data.qfrc_actuator[0].copy()
print("Final torque at 20deg with BRA act=0.1:", final_torque)

#record.save_video(record.render_frames(model, states, IMG_HEIGHT, IMG_WIDTH, camera="side_view", time_series=elbow_angle_series, plot_title="Elbow Angle [degrees]"), "muscle_Pctrl_elbow", FPS)
record.plot_data(elbow_angle_series, "muscle_torque_map/20deg/elbow_angle", title="Elbow Angle [degrees]")
record.plot_data(muscle_activation_series, "muscle_torque_map/20deg/muscle_activation", title="BRA Muscle Activation")
record.plot_data(torque_series, "muscle_torque_map/20deg/torque", title="BRA act=0.1 --> Torque [Nm]")

