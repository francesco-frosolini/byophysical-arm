# Open loop control for elbow using one actuator's input from CSV
import sys
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0
import myosuite
from myosuite.utils import gym
import mujoco as m
import numpy as np
sys.modules['numpy'] = np  

import record
import pandas as pd

IMG_HEIGHT = 1088
IMG_WIDTH = 1088
SIMLEN = 1.2  # seconds
FPS = 30
SEED = 1
TIMESTEP = 0.002

np.random.seed(SEED)

env: PoseEnvV0 = gym.make('myoElbowPose1D6MFixed-v0')  # type: ignore #unclear if type hinting is a good idea
_, _ = env.reset(seed=SEED)

env.unwrapped.target_type = 'fixed'
trgAngle = 90  # degrees
env.unwrapped.target_jnt_value = np.deg2rad(trgAngle)
obs, _ = env.reset(seed=SEED)

mujoco_model: m.MjModel = env.unwrapped.sim.model.ptr
mujoco_data: m.MjData = env.unwrapped.sim.data.ptr

start_angle = np.deg2rad(20)
mujoco_data.qpos[0] = start_angle
mujoco_data.qvel[0] = 0.0
# get info about the body's actuators
actuator_objs = [mujoco_model.actuator(i) for i in range(6)]
actuator_names = [actuator.name for actuator in actuator_objs]

m.mj_forward(mujoco_model, mujoco_data)

simstart = mujoco_data.time

states = []
state_size = m.mj_stateSize(mujoco_model, m.mjtState.mjSTATE_INTEGRATION)

mujoco_model.vis.global_.offwidth = IMG_WIDTH
mujoco_model.vis.global_.offheight = IMG_HEIGHT


# Load one actuator's input file
i = 5  # actuator index
csv_file = f'plots/policy/actuator_input_{i}.csv'
df = pd.read_csv(csv_file)

#values = df[actuator_names[i]].values
values = np.linspace(0, 1, int(SIMLEN/TIMESTEP))  
#values = 0.5 * (1 + np.sin(2 * np.pi * np.arange(len(values)) / len(values) - np.pi / 2))
# obs, _ = env.reset(seed=SEED)
elbow_angle_series = []
actuator_force_series = []
actuator_input_series = []

step_count = 0
while (mujoco_data.time - simstart) < SIMLEN:
    # Set controls: only the i-th actuator from CSV, others to 0
    mujoco_data.ctrl[:] = 1.0
    #if step_count < len(values):
     #   mujoco_data.ctrl[i] = values[step_count]

    # Step the simulation
    m.mj_step(mujoco_model, mujoco_data)

    # Save state every 1/FPS seconds
    sensord = np.rad2deg(mujoco_data.qpos[0].copy())
    actuator_force = mujoco_data.actuator_force.copy()
    actuator_input = mujoco_data.ctrl.copy()
    elbow_angle_series.append((mujoco_data.time, sensord))
    actuator_force_series.append((mujoco_data.time, actuator_force))
    actuator_input_series.append((mujoco_data.time, actuator_input))

    if len(states) < (mujoco_data.time - simstart) * FPS:
        state_buffer = np.empty(state_size, dtype=np.float64)
        m.mj_getState(mujoco_model, mujoco_data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
        states.append(state_buffer.copy())

    step_count += 1

#record.save_video(record.render_frames(mujoco_model, states, IMG_HEIGHT, IMG_WIDTH, camera="side_view", time_series=elbow_angle_series, plot_title="Elbow Angle [degrees]"), "ol_elbow", FPS)
record.plot_data(elbow_angle_series, "ol_ctrl/ol_elbow_angle")

for j in range(len(actuator_force_series[0][1])):
    force_series_j = [(t, forces[j]) for t, forces in actuator_force_series]
    record.plot_data(force_series_j, f"ol_ctrl/ol_actuator_force_{j}", actuator_names[j])

for j in range(len(actuator_input_series[0][1])):
    input_series_j = [(t, inputs[j]) for t, inputs in actuator_input_series]
    record.plot_data(input_series_j, f"ol_ctrl/ol_actuator_input_{j}", actuator_names[j])
