import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumContiEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum_conti.xml', 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        p, theta1, pdot, theta1dot = ob[0], ob[1], ob[2], ob[3]
        tip_x = p + 0.6 * np.sin(theta1)
        tip_y = 0.6 * np.cos(theta1)
        dist_penalty = 0.01 * np.power(tip_x, 2) + 0.04*np.power(tip_y - 0.6, 2)
        vel_penalty = 1e-3 * np.power(pdot, 2) + 1e-3 * np.power(theta1dot, 2)
        reward = -dist_penalty - vel_penalty
        notdone = (np.abs(p) <= 1.) and (np.abs(theta1) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


def test_reset():
    from utils.dummy_vec_env import DummyVecEnv
    env = InvertedPendulumContiEnv()
    obs = env.reset()
    wrapenv = DummyVecEnv(env)
    for i in range(10):
        wrapenv.reset(init_obs=np.array([obs]))
        wrapenv.render()
        for j in range(100):
            action = np.array([wrapenv.action_space.sample()])
            wrapenv.step(action)
            wrapenv.render()


if __name__ == '__main__':
    test_reset()
