# Open loop control for elbow using data from plant_data.json

import json
import sys
from zipfile import Path
import mujoco as m
import numpy as np
from sympy import to_cnf

from plant_models import PlantPlotData
sys.modules['numpy'] = np  
import time

import record
import pandas as pd

IMG_HEIGHT = 1088
IMG_WIDTH = 1088
SIMLEN = 10  # seconds
FPS = 30
SEED = 1
TIMESTEP = 0.001

np.random.seed(SEED)



model = m.MjModel.from_xml_path("models/myo_motor/myoelbow_1dof1act.xml")
model.opt.timestep = TIMESTEP
data = m.MjData(model)


simstart = data.time

states = []
state_size = m.mj_stateSize(model, m.mjtState.mjSTATE_INTEGRATION)

model.vis.global_.offwidth = IMG_WIDTH
model.vis.global_.offheight = IMG_HEIGHT
elbow_angle_series = []


start= time.time()
step_count = 0

path = "./plant_data.json"
with open(path, "r") as f:
    json = f.read()
plant_data = PlantPlotData.model_validate_json(json)

#list_torques = plant_data.joint_data[1].input_cmd_torque
list_torques = np.linspace(-1, 1, int(SIMLEN/TIMESTEP))
motor_activation_series = []


while (data.time - simstart) < SIMLEN:
    # Save state every 1/FPS seconds
    data.ctrl[0] = 0.2
    #if step_count < len(list_torques):
    #    data.ctrl[0] = list_torques[step_count]
    
    
    sensord = np.rad2deg(data.qpos[0].copy())
    elbow_angle_series.append((data.time, sensord))
    motor_activation_series.append((data.time, data.ctrl[0].copy()))

    if len(states) < (data.time - simstart) * FPS:
        state_buffer = np.empty(state_size, dtype=np.float64)
        m.mj_getState(model, data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
        states.append(state_buffer.copy())


    # Step the simulation
    m.mj_step(model, data)
    step_count += 1
    #print(f"Simulation time: {time.time() - start} seconds")


#record.save_video(record.render_frames(model, states, IMG_HEIGHT, IMG_WIDTH, camera="side_view", time_series=elbow_angle_series, plot_title="Elbow Angle [degrees]"), "ol_1act_elbow", FPS)
record.plot_data(elbow_angle_series, "1act_ol_ctrl/ol_elbow_angle")
record.plot_data(motor_activation_series, "1act_ol_ctrl/motor_activation", title="Motor Activation [-1 to 1]")