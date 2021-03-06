import gym
import numpy as np
from gym.utils import seeding
import matplotlib.pyplot as plt

class UpperTriangle(gym.Env):
    def __init__(self, **kwargs):
        metadata = {'render.modes': ['human']}
        self.step_length = 0.1  # ms
        self.action_number = 1
        self.action_space = gym.spaces.Box(low=-0.5, high=0.5, shape=(self.action_number,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.full([2,], -float('inf')),np.full([2,], float('inf')), dtype=np.float32)
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
        self.obs = (np.matmul(self.A, self.obs[:, np.newaxis]) + np.matmul(self.B, self.action[:, np.newaxis])).reshape([-1,])
        constraint = np.max(np.abs(self.obs)) - 5.
        done = True if constraint > 0 else False
        info = dict(reward_info=dict(reward=reward, constraints=float(constraint)))
        self.cstr = constraint
        return self.obs, reward, done, info # s' r



    def _action_transform(self, action):
        action = 0.5 * np.clip(action, -1.05, 1.05)
        return action

    def compute_reward(self, obs, action):
        r = obs[0] ** 2 + obs[1] ** 2 + 10. * action[0] ** 2
        return r

    def _reset_init_state(self):
        d = -10. * np.random.random() + 5
        v = -10. * np.random.random() + 5
        return np.array([d,v])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        extension = 1
        if mode == 'human':
            plt.ion()
            plt.cla()
            plt.title("Upper Triangle")
            ax = plt.axes(xlim=(-5 - extension, 5 + extension),
                          ylim=(-5 - extension, 5 + extension))
            plt.axis("equal")
            plt.axis('off')

            ax.add_patch(plt.Rectangle((-5, -5), 10, 10, edgecolor='black', facecolor='none'))
            plt.scatter(self.obs[0], self.obs[1])

            text_x, text_y = -5, 5
            plt.text(text_x, text_y, 'x: {:.2f}'.format(self.obs[0]))
            plt.text(text_x, text_y - 1, 'y: {:.2f}'.format(self.obs[1]))
            plt.text(text_x, text_y - 2, 'action: {:.2f}'.format(self.action[0]))
            plt.text(text_x, text_y - 3, 'constraints: {:.2f}'.format(self.cstr))
            plt.show()
            plt.pause(0.001)

def env():
    import time
    env = UpperTriangle()
    env.step_length = 0.01
    obs = env.reset()
    action = np.array([0.5])
    while True:
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.5)
        if done: env.reset()
        print(reward)

if __name__ == '__main__':
    env()










