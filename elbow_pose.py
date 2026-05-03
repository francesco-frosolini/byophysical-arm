#make the elbow reach a pose using pretrained policy and gym environment
import sys
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0
import myosuite
from myosuite.utils import gym
import mujoco as m
import numpy as np
sys.modules['numpy'] = np #needed for pickle loading of the policy, which depends on numpy.
import pickle

import record
import pandas as pd

IMG_HEIGHT = 1088
IMG_WIDTH = 1088
SIMLEN=1.2 #seconds
FPS=30
SEED = 1

np.random.seed(SEED)

env : PoseEnvV0 = gym.make('myoElbowPose1D6MFixed-v0') #type: ignore #unclear if type hinting is a good idea
_, _ = env.reset(seed=SEED)
policy=".venv/lib/python3.10/site-packages/myosuite/agents/baslines_NPG/myoElbowPose1D6MRandom-v0/2022-02-26_21-16-27/35_env=myoElbowPose1D6MRandom-v0,seed=3/iterations/best_policy.pickle"
pi=pickle.load(open(policy, 'rb'))



env.unwrapped.target_type = 'fixed'
trgAngle= 90 #degrees
env.unwrapped.target_jnt_value = np.deg2rad(trgAngle)
obs, _ = env.reset(seed=SEED)

mujoco_model : m.MjModel = env.unwrapped.sim.model.ptr
mujoco_data : m.MjData = env.unwrapped.sim.data.ptr

start_angle = np.deg2rad(20)
mujoco_data.qpos[0] = start_angle 
mujoco_data.qvel[0] = 0.0  
#get info about the body's acutators
actuator_objs = [mujoco_model.actuator(i) for i in range(6)]        
actuator_names = [actuator.name for actuator in actuator_objs]

m.mj_forward(mujoco_model, mujoco_data) 

simstart = mujoco_data.time

states = []
state_size = m.mj_stateSize(mujoco_model, m.mjtState.mjSTATE_INTEGRATION)
#mujoco_model.opt.timestep = 0.001
mujoco_model.vis.global_.offwidth = IMG_WIDTH
mujoco_model.vis.global_.offheight = IMG_HEIGHT

#obs, _ = env.reset(seed=SEED)
elbow_angle_series = []
actuator_force_series = []
actuator_input_series = []

while (mujoco_data.time - simstart) < SIMLEN:

    o = env.unwrapped.get_obs() #type: ignore
    # get the next action from the policy
    action, _ = pi.get_action(o)
    # take an action based on the current observation
    obs, reward, done, _, info = env.step(action)
    # Save state every 1/FPS seconds
    sensord=np.rad2deg(mujoco_data.qpos[0].copy())
    actuator_force = mujoco_data.actuator_force.copy()
    actuator_input = mujoco_data.ctrl.copy() 
    elbow_angle_series.append((mujoco_data.time, sensord))
    actuator_force_series.append((mujoco_data.time, actuator_force))
    actuator_input_series.append((mujoco_data.time, actuator_input))

    if len(states) < (mujoco_data.time - simstart) * FPS:
        state_buffer = np.empty(state_size, dtype=np.float64)
        m.mj_getState(mujoco_model, mujoco_data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
        states.append(state_buffer.copy())


#record.save_video(record.render_frames(mujoco_model, states, IMG_HEIGHT, IMG_WIDTH, camera="side_view", time_series=elbow_angle_series, plot_title="Elbow Angle [degrees]"), "strike_elbow_pose", FPS)
# TODO create directory if it doesn't exist (inside record.py?)

record.plot_data(elbow_angle_series, "policy/elbow_angle")

for i in range(len(actuator_force_series[0][1])):
    force_series_i = [(t, forces[i]) for t, forces in actuator_force_series]
    record.plot_data(force_series_i, f"policy/actuator_force_{i}", actuator_names[i])

for i in range(len(actuator_input_series[0][1])):
    input_series_i = [(t, inputs[i]) for t, inputs in actuator_input_series]
    record.plot_data(input_series_i, f"policy/actuator_input_{i}",actuator_names[i]) 

actuator_5_input = [(t, inputs[5]) for t, inputs in actuator_input_series]

# CSV export
for i, name in enumerate(actuator_names):
    actuator_inputs_data = {
        'time': [t for t, _ in actuator_input_series],
        name: [inputs[i] for _, inputs in actuator_input_series]
    }
    df = pd.DataFrame(actuator_inputs_data)
    df.to_csv(f'plots/policy/actuator_input_{i}.csv', index=False)