import gymnasium as gym
from gymnasium import spaces
from sim_class import Simulation

class OT2Env(gym.Env):
    
    def __init__(self, num_agents=1):
        """
        Initialize the environment.

        Parameters:
        - num_agents (int): Number of OT-2 robots in the simulation.
        """
        self.sim = Simulation(num_agents=num_agents)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        self.observation_space = spaces.Dict({
            'joint_states': spaces.Dict({
                'joint_0': spaces.Box(low=-1, high=1, shape=(4,), dtype=float),
                'joint_1': spaces.Box(low=-1, high=1, shape=(4,), dtype=float),
                'joint_2': spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
            }),
            'robot_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=float),
            'pipette_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        })

    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
        - action (numpy.ndarray): Action vector.

        Returns:
        - observation (dict): Current observation.
        - reward (float): Reward for the current step.
        - done (bool): Whether the episode is done.
        - info (dict): Additional information.
        """
        # Scale the action values to match the expected range
        action = action * [0.1, 0.1, 0.1, 1.0]
        self.sim.run([action])
        obs = self.sim.get_states()
        reward = self.calculate_reward(obs)
        done = self.is_done(obs)
        return obs, reward, done, {}

    def reset(self):
        """
        Reset the environment.

        Returns:
        - observation (dict): Initial observation after reset.
        """
        self.sim.reset()
        return self.sim.get_states()

    def render(self, mode='human'):
        """
        Render the environment.

        Parameters:
        - mode (str): Rendering mode.

        Note: Rendering is handled in the Simulation class.
        """
        pass

    def close(self):
        """
        Close the environment.
        """
        self.sim.close()

    def calculate_reward(self, obs):
        """
        Calculate the reward for the current observation.

        Parameters:
        - obs (dict): Current observation.

        Returns:
        - reward (float): Reward value.
        """
        # Euclidean distance between current pipette position and target position
        target_position = [0.0, 0.0, 0.0] 
        current_position = obs['robotId_1']['pipette_position']
        distance = sum((x - y) ** 2 for x, y in zip(target_position, current_position)) ** 0.5
        return -distance  # Negative distance as it is a minimization problem

    def is_done(self, obs):
        """
        Check if the episode is done based on the current observation.

        Parameters:
        - obs (dict): Current observation.

        Returns:
        - done (bool): Whether the episode is done.
        """
        # Terminate when the robot reaches a certain height
        return obs['robotId_1']['pipette_position'][2] >= 0.2