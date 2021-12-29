import numpy as np
from gym.envs.registration import register

def register_custom_env():
        # finite time convergence test suite
    config = {
    'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point

    # finite time convergence test suite modification
    'robot_placements': None,  # Robot placements list (defaults to full extents)
    'robot_locations': [[0.0, 0.0]],  # Explicitly place robot XY coordinate
    'robot_keepout': 0.0,  # Needs to be set to match the robot XML used
    # Hazardous areas
    'hazards_placements': None,  # Placements list for hazards (defaults to full extents)
    'hazards_locations': [[-0.3, -0.3]],  # Fixed locations to override placements
    'hazards_keepout': 0.0,  # Radius of hazard keepout for placement
    'hazards_num': 1,
    'hazards_size': 0.5,

    'task': 'goal',
    'observation_flatten': True,  # Flatten observation into a vector
    'observe_sensors': True,  # Observe all sensor data from simulator
    # Sensor observations
    # Specify which sensors to add to observation space
    'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
    'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
    'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
    'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

    #observe goal/box/...
    'observe_goal_dist': False,  # Observe the distance to the goal
    'observe_goal_comp': False,  # Observe a compass vector to the goal
    'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
    'observe_box_comp': False,  # Observe the box with a compass
    'observe_box_lidar': False,  # Observe the box with a lidar
    'observe_circle': False,  # Observe the origin with a lidar
    'observe_remaining': False,  # Observe the fraction of steps remaining
    'observe_walls': False,  # Observe the walls with a lidar space
    'observe_hazards': True,  # Observe the vector from agent to hazards
    'observe_vases': True,  # Observe the vector from agent to vases
    'observe_pillars': False,  # Lidar observation of pillar object positions
    'observe_buttons': False,  # Lidar observation of button object positions
    'observe_gremlins': False,  # Gremlins are observed with lidar-like space
    'observe_vision': False,  # Observe vision from the robot

    # Constraints - flags which can be turned on
    # By default, no constraints are enabled, and all costs are indicator functions.
    'constrain_hazards': True,  # Constrain robot from being in hazardous areas
    'constrain_vases': False,  # Constrain frobot from touching objects
    'constrain_pillars': False,  # Immovable obstacles in the environment
    'constrain_buttons': False,  # Penalize pressing incorrect buttons
    'constrain_gremlins': False,  # Moving objects that must be avoided
    # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
    'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

    #lidar setting
    'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
    'lidar_num_bins': 16,
    #num setting

    'vases_num': 0,

    # dt perhaps?

    # Frameskip is the number of physics simulation steps per environment step
    # Frameskip is sampled as a binomial distribution
    # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
    'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) # 经过验证，这个参数和xml的参数是等价的
    'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
}
    env_id = 'Safexp-CustomGoal1-v0'
    register(id=env_id,
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

    config = {
        'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
        'task': 'goal',
        'observation_flatten': True,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        #observe goal/box/...
        'observe_goal_dist': False,  # Observe the distance to the goal
        'observe_goal_comp': False,  # Observe a compass vector to the goal
        'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
        'observe_box_comp': False,  # Observe the box with a compass
        'observe_box_lidar': False,  # Observe the box with a lidar
        'observe_circle': False,  # Observe the origin with a lidar
        'observe_remaining': False,  # Observe the fraction of steps remaining
        'observe_walls': False,  # Observe the walls with a lidar space
        'observe_hazards': True,  # Observe the vector from agent to hazards
        'observe_vases': True,  # Observe the vector from agent to vases
        'observe_pillars': False,  # Lidar observation of pillar object positions
        'observe_buttons': False,  # Lidar observation of button object positions
        'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        'observe_vision': False,  # Observe vision from the robot

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': True,  # Constrain robot from being in hazardous areas
        'constrain_vases': False,  # Constrain frobot from touching objects
        'constrain_pillars': False,  # Immovable obstacles in the environment
        'constrain_buttons': False,  # Penalize pressing incorrect buttons
        'constrain_gremlins': False,  # Moving objects that must be avoided
        # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
        'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

        #lidar setting
        'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_num_bins': 16,
        #num setting
        'hazards_num': 8,
        'hazards_size': 0.45,
        'vases_num': 0,

        # dt perhaps?

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) # 经过验证，这个参数和xml的参数是等价的
        'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
    }
    env_id = 'Safexp-CustomGoal2-v0'
    register(id=env_id,
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

    config = config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': False,  # Observe the vector from agent to hazards
            'observe_vases': False,  # Observe the vector from agent to vases
            'observe_pillars': True,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': False,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': True,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 0,
            'hazards_size': 0.15,
            'vases_num': 0,

            # Pillars (immovable obstacles we should not touch)

            # 'robot_keepout': 0.4,

            'pillars_num': 8,  # Number of pillars in the world
            'pillars_placements': None,  # Pillars placements list (defaults to full extents)
            # 'pillars_locations': [],  # Fixed locations to override placements
            'pillars_keepout': 0.4,  # Radius for placement of pillars
            'pillars_size': 0.30,  # Half-size (radius) of pillar objects
            'pillars_height': 0.5,  # Half-height of pillars geoms
            'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar


            # dt perhaps?

            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) # 经过验证，这个参数和xml的参数是等价的
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }
    env_id = 'Safexp-CustomGoalPillar2-v0'
    register(id=env_id,
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

    config = config = {
            'robot_base': 'xmls/point.xml', # dt in xml, default 0.002s for point
            'task': 'goal',
            'observation_flatten': True,  # Flatten observation into a vector
            'observe_sensors': True,  # Observe all sensor data from simulator
            # Sensor observations
            # Specify which sensors to add to observation space
            'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
            'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
            'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
            'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

            #observe goal/box/...
            'observe_goal_dist': False,  # Observe the distance to the goal
            'observe_goal_comp': False,  # Observe a compass vector to the goal
            'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
            'observe_box_comp': False,  # Observe the box with a compass
            'observe_box_lidar': False,  # Observe the box with a lidar
            'observe_circle': False,  # Observe the origin with a lidar
            'observe_remaining': False,  # Observe the fraction of steps remaining
            'observe_walls': False,  # Observe the walls with a lidar space
            'observe_hazards': False,  # Observe the vector from agent to hazards
            'observe_vases': False,  # Observe the vector from agent to vases
            'observe_pillars': True,  # Lidar observation of pillar object positions
            'observe_buttons': False,  # Lidar observation of button object positions
            'observe_gremlins': False,  # Gremlins are observed with lidar-like space
            'observe_vision': False,  # Observe vision from the robot

            # Constraints - flags which can be turned on
            # By default, no constraints are enabled, and all costs are indicator functions.
            'constrain_hazards': False,  # Constrain robot from being in hazardous areas
            'constrain_vases': False,  # Constrain frobot from touching objects
            'constrain_pillars': True,  # Immovable obstacles in the environment
            'constrain_buttons': False,  # Penalize pressing incorrect buttons
            'constrain_gremlins': False,  # Moving objects that must be avoided
            # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
            'constrain_indicator': False,  # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

            #lidar setting
            'lidar_max_dist': None, # Maximum distance for lidar sensitivity (if None, exponential distance)
            'lidar_num_bins': 16,
            #num setting
            'hazards_num': 0,
            'hazards_size': 0.15,
            'vases_num': 0,

            # Pillars (immovable obstacles we should not touch)

            # 'robot_keepout': 0.4,

            'pillars_num': 8,  # Number of pillars in the world
            'pillars_placements': None,  # Pillars placements list (defaults to full extents)
            # 'pillars_locations': [],  # Fixed locations to override placements
            'pillars_keepout': 0.4,  # Radius for placement of pillars
            'pillars_size': 0.45,  # Half-size (radius) of pillar objects
            'pillars_height': 0.5,  # Half-height of pillars geoms
            'pillars_cost': 1.0,  # Cost (per step) for being in contact with a pillar


            # dt perhaps?

            # Frameskip is the number of physics simulation steps per environment step
            # Frameskip is sampled as a binomial distribution
            # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
            'frameskip_binom_n': 10,  # Number of draws trials in binomial distribution (max frameskip) # 经过验证，这个参数和xml的参数是等价的
            'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
        }
    env_id = 'Safexp-CustomGoalPillar3-v0'
    register(id=env_id,
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

    config = {
        'robot_base': 'xmls/point.xml',  # dt in xml, default 0.002s for point
        'task': 'push',
        'box_size': 0.2,
        'box_null_dist': 0,

        'observation_flatten': True,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        # observe goal/box/...
        'observe_goal_dist': False,  # Observe the distance to the goal
        'observe_goal_comp': False,  # Observe a compass vector to the goal
        'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
        'observe_box_comp': False,  # Observe the box with a compass
        'observe_box_lidar': True,  # Observe the box with a lidar
        'observe_circle': False,  # Observe the origin with a lidar
        'observe_remaining': False,  # Observe the fraction of steps remaining
        'observe_walls': False,  # Observe the walls with a lidar space
        'observe_hazards': True,  # Observe the vector from agent to hazards
        'observe_vases': True,  # Observe the vector from agent to vases
        'observe_pillars': False,  # Lidar observation of pillar object positions
        'observe_buttons': False,  # Lidar observation of button object positions
        'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        'observe_vision': False,  # Observe vision from the robot

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': True,  # Constrain robot from being in hazardous areas
        'constrain_vases': False,  # Constrain frobot from touching objects
        'constrain_pillars': False,  # Immovable obstacles in the environment
        'constrain_buttons': False,  # Penalize pressing incorrect buttons
        'constrain_gremlins': False,  # Moving objects that must be avoided
        # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
        'constrain_indicator': False,
        # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

        # lidar setting
        'lidar_max_dist': None,  # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_num_bins': 16,
        # num setting
        'hazards_num': 1,
        'hazards_size': 0.15,
        'vases_num': 0,

        # dt perhaps?

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,
        # Number of draws trials in binomial distribution (max frameskip) # 经过验证，这个参数和xml的参数是等价的
        'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
    }
    env_id = 'Safexp-CustomPush1-v0'
    register(id=env_id,
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

    config = {
        'robot_base': 'xmls/point.xml',  # dt in xml, default 0.002s for point
        'task': 'push',
        'box_size': 0.2,
        'box_null_dist': 0,

        'observation_flatten': True,  # Flatten observation into a vector
        'observe_sensors': True,  # Observe all sensor data from simulator
        # Sensor observations
        # Specify which sensors to add to observation space
        'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
        'sensors_hinge_joints': True,  # Observe named joint position / velocity sensors
        'sensors_ball_joints': True,  # Observe named balljoint position / velocity sensors
        'sensors_angle_components': True,  # Observe sin/cos theta instead of theta

        # observe goal/box/...
        'observe_goal_dist': False,  # Observe the distance to the goal
        'observe_goal_comp': False,  # Observe a compass vector to the goal
        'observe_goal_lidar': True,  # Observe the goal with a lidar sensor
        'observe_box_comp': False,  # Observe the box with a compass
        'observe_box_lidar': True,  # Observe the box with a lidar
        'observe_circle': False,  # Observe the origin with a lidar
        'observe_remaining': False,  # Observe the fraction of steps remaining
        'observe_walls': False,  # Observe the walls with a lidar space
        'observe_hazards': True,  # Observe the vector from agent to hazards
        'observe_vases': True,  # Observe the vector from agent to vases
        'observe_pillars': False,  # Lidar observation of pillar object positions
        'observe_buttons': False,  # Lidar observation of button object positions
        'observe_gremlins': False,  # Gremlins are observed with lidar-like space
        'observe_vision': False,  # Observe vision from the robot

        # Constraints - flags which can be turned on
        # By default, no constraints are enabled, and all costs are indicator functions.
        'constrain_hazards': True,  # Constrain robot from being in hazardous areas
        'constrain_vases': False,  # Constrain frobot from touching objects
        'constrain_pillars': False,  # Immovable obstacles in the environment
        'constrain_buttons': False,  # Penalize pressing incorrect buttons
        'constrain_gremlins': False,  # Moving objects that must be avoided
        # cost discrete/continuous. As for AdamBA, I guess continuous cost is more suitable.
        'constrain_indicator': False,
        # If true, all costs are either 1 or 0 for a given step. If false, then we get dense cost.

        # lidar setting
        'lidar_max_dist': None,  # Maximum distance for lidar sensitivity (if None, exponential distance)
        'lidar_num_bins': 16,
        # num setting
        'hazards_num': 8,
        'hazards_size': 0.30,
        'vases_num': 0,

        # dt perhaps?

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        'frameskip_binom_n': 10,
        # Number of draws trials in binomial distribution (max frameskip) # 经过验证，这个参数和xml的参数是等价的
        'frameskip_binom_p': 1.0  # Probability of trial return (controls distribution)
    }
    env_id = 'Safexp-CustomPush2-v0'
    register(id=env_id,
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})

if __name__ == '__main__':
    import gym
    import safety_gym
    import time
    register_custom_env()
    env = gym.make('Safexp-CustomGoal1-v0')
    env.reset()
    action = np.array([1.,0.])
    while True:
        _, _, done, info = env.step(action)
        print(info.get('dist'))
        env.render()
        time.sleep(0.1)
        if done:
            env.reset()
