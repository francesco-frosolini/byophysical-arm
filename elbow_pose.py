#make the elbow reach a pose using pretrained policy and gym environment
import sys
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0
import myosuite
from myosuite.utils import gym
import mujoco as m
import numpy as np
sys.modules['numpy'] = np #needed for pickle loading of the policy, which depends on numpy.
import pickle

from tqdm import tqdm
import record

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

mujoco_model = env.unwrapped.sim.model.ptr
mujoco_data = env.unwrapped.sim.data.ptr

start_angle = np.deg2rad(20)
mujoco_data.qpos[0] = start_angle 
mujoco_data.qvel[0] = 0.0          

m.mj_forward(mujoco_model, mujoco_data) 

simstart = mujoco_data.time

states = []
state_size = m.mj_stateSize(mujoco_model, m.mjtState.mjSTATE_INTEGRATION)
#mujoco_model.opt.timestep = 0.001
mujoco_model.vis.global_.offwidth = IMG_WIDTH
mujoco_model.vis.global_.offheight = IMG_HEIGHT

#obs, _ = env.reset(seed=SEED)
elbow_angle_series = []


while (mujoco_data.time - simstart) < SIMLEN:

    o = env.unwrapped.get_obs() #type: ignore
    # get the next action from the policy
    action, _ = pi.get_action(o)
    # take an action based on the current observation
    obs, reward, done, _, info = env.step(action)
    # Save state every 1/FPS seconds
    sensord=np.rad2deg(mujoco_data.qpos[0].copy())
    elbow_angle_series.append((mujoco_data.time, sensord))


    if len(states) < (mujoco_data.time - simstart) * FPS:
        state_buffer = np.empty(state_size, dtype=np.float64)
        m.mj_getState(mujoco_model, mujoco_data, state_buffer, m.mjtState.mjSTATE_INTEGRATION)
        states.append(state_buffer.copy())


#record.save_video(record.render_frames(mujoco_model, states, IMG_HEIGHT, IMG_WIDTH, camera="side_view", time_series=elbow_angle_series, plot_title="Elbow Angle [degrees]"), "strike_elbow_pose", FPS)
record.plot_data(elbow_angle_series, "elbow_angle")
