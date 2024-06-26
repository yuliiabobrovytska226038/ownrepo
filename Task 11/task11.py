import os
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pandas as pd
import pybullet_data
import time
import wandb
import math
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
from wasabi import Printer
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path
import tempfile
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecVideoRecorder,
)


msg = Printer()

logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)


class PipetteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PipetteEnv, self).__init__()

        # Connect to PyBullet
        self.client = p.connect(p.GUI)#p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load the simulation environment
        self.planeId = p.loadURDF("plane.urdf")
        self.startPos = [0, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, -1.57])
        self.OT2Id = p.loadURDF("Y2B-2023-OT2_Twin/ot_2_simulation_v6.urdf", self.startPos, self.startOrientation, useFixedBase=True)

        # Set the camera parameters
        num_agents = 1
        cameraDistance = 1.1 * (math.ceil((num_agents)**0.3))  # Distance from the target (zoom)
        cameraYaw = 0  # Rotation around the vertical axis in degrees
        cameraPitch = -35  # Rotation around the horizontal axis in degrees
        cameraTargetPosition = [0, -(math.ceil(num_agents**0.5)/2)+0.5, 0.1]  # XYZ coordinates of the target position
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        # Define the joint indices
        self.x_joint_index = 1  # index for gantry_x1
        self.y_joint_index = 0  # index for gantry_y1
        self.z_joint_index = 2  # index for gantry_z1

        # Get joint limits
        self.x_min, self.x_max = self.get_joint_limits(self.OT2Id, self.x_joint_index)
        self.y_min, self.y_max = self.get_joint_limits(self.OT2Id, self.y_joint_index)
        self.z_min, self.z_max = self.get_joint_limits(self.OT2Id, self.z_joint_index)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-0.1, -0.1, -0.1]), high=np.array([0.1, 0.1, 0.1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([self.x_min, self.y_min, self.z_min]), high=np.array([self.x_max, self.y_max, self.z_max]), dtype=np.float32)

        # Define initial and goal positions
        self.initial_position = np.array([(self.x_max + self.x_min) / 2, (self.y_max + self.y_min) / 2, (self.z_max + self.z_min) / 2])
        self.goal_position = None
        self.current_position = self.initial_position.copy()

        self.done = False

    def get_joint_limits(self, robot_id, joint_index):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]
        return joint_lower_limit, joint_upper_limit

    def get_end_effector_position(self, robot_id):
        end_effector_state = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1)
        return end_effector_state[4]

    def _move_to(self, position):
        p.setJointMotorControl2(bodyUniqueId=self.OT2Id, jointIndex=self.x_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=position[0])
        p.setJointMotorControl2(bodyUniqueId=self.OT2Id, jointIndex=self.y_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=position[1])
        p.setJointMotorControl2(bodyUniqueId=self.OT2Id, jointIndex=self.z_joint_index, controlMode=p.POSITION_CONTROL, targetPosition=position[2])
        p.stepSimulation()
        time.sleep(1./240.)  # simulation step time

    def reset(self):
        self.goal_position = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.current_position = self.initial_position.copy()
        self.done = False
        p.resetSimulation(self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        self.OT2Id = p.loadURDF("Y2B-2023-OT2_Twin/ot_2_simulation_v6.urdf", self.startPos, self.startOrientation, useFixedBase=True)
        self._move_to(self.current_position)
        return self.current_position

    def step(self, action):
        self.current_position += action
        self.current_position = np.clip(self.current_position, self.observation_space.low, self.observation_space.high)
        self._move_to(self.current_position)

        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        reward = -distance_to_goal
        print(distance_to_goal)
        self.done = distance_to_goal < 0.1#0.01

        return self.current_position, reward, self.done, {},distance_to_goal

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=1.5,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

    def close(self):
        p.disconnect(self.client)

# Define the working envelope
X_min, X_max = -0.26, 0.13
Y_min, Y_max = -0.18, 0.26
Z_min, Z_max = 0.05, 0.17

# Register the environment
gym.envs.registration.register(
    id='PipetteEnv-v1',
    entry_point='__main__:PipetteEnv',
)

env_id = "PipetteEnv-v1"
env = PipetteEnv()

logs =[] #logs
models_dir = 'model'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model = PPO('MlpPolicy', env, n_steps=1000, device="cuda", verbose=0)


##training the model
TIMESTEPS = 10000
iters = 0
for _ in range(1000):
  iters += 1
  model = model = PPO(
      policy = 'MlpPolicy',
      env = env,
      n_steps = 2048,
      batch_size = 128,
      n_epochs = 6,
      gamma = 0.999,
      device="cuda",
      gae_lambda = 0.98,
      ent_coef = 0.01,
      verbose=1)
  model.learn(total_timesteps=10000)
  print(iters)
  #mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
  #print(mean_reward)
  iters_n = str(iters)
  #model.save(f"model/ppo-opentrons-v3-{iters_n}")
  model.save(f"model/{TIMESTEPS*iters}")


##Finding the best model
model_list = os.listdir('model')
model = PPO.load(f"model/{model_list[-1]}",env=env)

##episodes
episodes = 4
records = []

for x in model_list:
  model = PPO.load(f"model/{x}",env=env)
  print("iteration")
  print(x)
  for ep in range(episodes):
    obs = env.reset()
    done = False
    score = 0
    while not done:
      action, _states = model.predict(obs)
      obs, reward, done, info,distance = env.step(action)
      '''
      print("**iter**")
      print(done)
      print(distance)
      print("**eval**")
      '''
      score += reward

  records.append([score, x])


reward_per_model = pd.DataFrame(records,columns=['reward','model'])
reward_per_model.to_csv('reward_per_model.csv')
reward_per_model.to_csv('reward_per_model.csv')
df = reward_per_model.sort_values(by='reward', ascending=False)
print(df)


df['reward'].plot(kind='hist', bins=20, title='reward')
plt.gca().spines[['top', 'right',]].set_visible(False)