import gym
# import gym_reachability
import numpy as np
import time
from gym.envs.registration import register

# register(
#     id="lunar_lander_reach-v0",
#     entry_point="gym_reachability.gym_reachability.envs:LunarLanderReachability"
# )


def try_gym_reach():
    env = gym.make("lunar_lander_reach-v0")
    obs = env.reset()
    print("reset obs: {}".format(obs))
    action = np.array([0,0])
    done = False
    while True:
        o, r, done, info = env.step(action)
        env.render()
        # print(o)
        print("reward: {:.2f}, cstr: {:.2f}".format(r, info.get('cost', 0)))
        # print("fly: {:.2f}, land: {:.2f}".format(info.get('flying_distance', 0), info.get('landing_distance', 0)))
        time.sleep(0.1)
        if done:
            print('DONE')
            time.sleep(3)
            env.reset()

from gym_reachability.gym_reachability.envs.dubins_car import DubinsCarEnv

def try_dubin_car():
    env = DubinsCarEnv()
    obs = env.reset()
    print("reset obs: {}".format(obs))
    action = 0
    done = False
    while True:
        o, r, done, info = env.step(action)
        env.render()
        # print(o)
        # print("reward: {:.2f}, cstr: {:.2f}".format(r, info.get('cost', 0)))
        # print("fly: {:.2f}, land: {:.2f}".format(info.get('flying_distance', 0), info.get('landing_distance', 0)))
        time.sleep(0.1)
        if done:
            print('DONE')
            time.sleep(3)
            env.reset()


if __name__ == '__main__':
    try_dubin_car()