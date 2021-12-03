import gym
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

class Air3d(gym.Env):
    def __init__(self, **kwargs):
        metadata = {'render.modes': ['human']}
        self.step_length = 0.1  # ms
        self.action_number = 1
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.action_number,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.full([3,], -float('inf')),np.full([3,], float('inf')), dtype=np.float32)
        self.obs = self._reset_init_state()
        self.A = np.array([[1., 1.],[0, 1.]])
        self.B = np.array([[0],[1.]])


    def reset(self):
        self.obs = self._reset_init_state()
        self.action = np.array([0.0])
        self.cstr = 0.0
        return self.obs

    def step(self, action):
        if len(action.shape) == 2:
            action = action.reshape([-1,])
        self.action = self._action_transform(action)
        reward = self.compute_reward(self.obs, self.action)
        dx = np.array([- 5. + 5. * np.cos(self.obs[2]) + action[0] * self.obs[1],
                             5. * np.sin(self.obs[2]) - action[0] * self.obs[0],
                             1. - action[0]
                             ])
        self.obs = self.obs + 0.1 * dx
        self.obs[2] = self.obs[2] % (2 * np.pi)
        constraint = 5 - np.linalg.norm(self.obs[:2])
        done = True if constraint > 0 else False
        info = dict(reward_info=dict(reward=reward, constraints=float(constraint)))
        self.cstr = constraint
        return self.obs, reward, done, info # s' r



    def _action_transform(self, action):
        action = 1. * np.clip(action, -1.05, 1.05)
        return action

    def compute_reward(self, obs, action):
        r = action[0] ** 2
        return r

    def _reset_init_state(self):
        x = -26. * np.random.random() + 20
        y = -20. * np.random.random() + 10
        phi = 2 * np.pi * np.random.random()
        return np.array([x, y, phi])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        extension = 1
        if mode == 'human':
            plt.ion()
            plt.cla()
            plt.title("Air3d")
            ax = plt.axes(xlim=(-6 - extension, 20 + extension),
                          ylim=(-10 - extension, 10 + extension))
            plt.axis("equal")
            plt.axis('off')

            ax.add_patch(plt.Rectangle((-6, -10), 26, 20, edgecolor='black', facecolor='none'))
            ax.add_patch(plt.Circle((0, 0), 5, edgecolor='red', facecolor='none'))
            plt.scatter(self.obs[0], self.obs[1])
            plt.arrow(self.obs[0], self.obs[1], 3 * np.cos(self.obs[2]), 3 * np.sin(self.obs[2]))

            text_x, text_y = -6, 10
            plt.text(text_x, text_y, 'x: {:.2f}'.format(self.obs[0]))
            plt.text(text_x, text_y - 1, 'y: {:.2f}'.format(self.obs[1]))
            plt.text(text_x, text_y - 2, 'angle: {:.2f}'.format(self.obs[2]))
            plt.text(text_x, text_y - 3, 'action: {:.2f}'.format(self.action[0]))
            plt.text(text_x, text_y - 4, 'constraints: {:.2f}'.format(self.cstr))
            plt.show()
            plt.pause(0.001)

def env():
    import time
    env = Air3d()
    # env.step_length = 0.01
    obs = env.reset()
    action = np.array([0.5])
    while True:
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)
        if done: env.reset()

if __name__ == '__main__':
    env()










