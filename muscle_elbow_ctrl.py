# Open loop muscular control for elbow

import json
import sys
import mujoco as m
import numpy as np
from simple_pid import PID

from plant_models import PlantPlotData
sys.modules['numpy'] = np  
import time

import record
import pandas as pd

IMG_HEIGHT = 1088
IMG_WIDTH = 1088
SIMLEN = 5  # seconds
FPS = 30
SEED = 1
TIMESTEP = 0.001

np.random.seed(SEED)


model = m.MjModel.from_xml_path("models/myo_sim/elbow/myoelbow_1dof6muscles.xml")
model.opt.timestep = TIMESTEP
data = m.MjData(model)


simstart = data.time

states = []
state_size = m.mj_stateSize(model, m.mjtState.mjSTATE_INTEGRATION)

model.vis.global_.offwidth = IMG_WIDTH
model.vis.global_.offheight = IMG_HEIGHT
elbow_angle_series = []
elbow_ref_series = []


start= time.time()
step_count = 0

path = "./plant_data.json"
with open(path, "r") as f:
    json = f.read()
plant_data = PlantPlotData.model_validate_json(json)

list_torques = plant_data.joint_data[1].input_cmd_torque

act_BRA = model.actuator("BRA")# controller

kp=0.01
ki=0.4
kd=0.00001

set_point=90.0  # degrees
controller=PID(kp,ki,kd,setpoint=set_point)
controller.sample_time = TIMESTEP
controller.output_limits = act_BRA.ctrlrange.copy() #for anti WU
muscle_activation_series = []


# sim loop
while (data.time - simstart) < SIMLEN:

    # apply ctrl
    sensord = np.rad2deg(data.qpos[0].copy())
    data.ctrl[act_BRA.id] = controller(sensord)

    elbow_angle_series.append((data.time, sensord))
    muscle_activation_series.append((data.time, data.ctrl[act_BRA.id]))
    elbow_ref_series.append((data.time, set_point))
    # Save state every 1/FPS seconds for video rendering
    if len(states) < (data.time - simstart) * FPS:
        state_buffer = np.empty(state_size, dtype=np.float64)
        m.mj_getState(model, data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
        states.append(state_buffer.copy())


    # Step the simulation
    m.mj_step(model, data)
    step_count += 1
    #print(f"Simulation time: {time.time() - start} seconds")


#record.save_video(record.render_frames(model, states, IMG_HEIGHT, IMG_WIDTH, camera="side_view", time_series=elbow_angle_series, plot_title="Elbow Angle [degrees]"), "muscle_Pctrl_elbow", FPS)
record.plot_data(elbow_angle_series, "muscle_P_ctrl/elbow_angle", title="Elbow Angle [degrees]", ref_series=elbow_ref_series)
record.plot_data(muscle_activation_series, "muscle_P_ctrl/BRA_activation", title="BRA Muscle Activation")
