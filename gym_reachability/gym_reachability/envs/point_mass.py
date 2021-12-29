# Copyright (c) 2020–2021, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Vicenc Rubies-Royo   ( vrubies@berkeley.edu )

import gym.spaces
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt

from utils import nearest_real_grid_point
from utils import visualize_matrix
from utils import q_values_from_q_func
from utils import state_to_index
from utils import index_to_state
from utils import v_from_q

# matplotlib.use("TkAgg")
matplotlib.style.use('ggplot')


class PointMassEnv(gym.Env):

    def __init__(self):

        # State bounds.
        self.bounds = np.array([[-1.9, 1.9],  # axis_0 = state, axis_1 = bounds.
                                [-2, 9.25]])
        # self.bounds = np.array([[-10, 10],  # axis_0 = state, axis_1 = bounds.
        #                         [-10, 10]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.05

        # Dubins car parameters.
        self.upward_speed = 2.0

        # Control parameters.
        self.horizontal_rate = 1
        self.discrete_controls = np.array([-self.horizontal_rate,
                                           0,
                                           self.horizontal_rate])

        # Constraint set parameters.
        # X,Y position and Side Length.
        self.box1_x_y_length = np.array([1.25, 2, 1.5])  # Bottom right.
        self.corners1 = np.array([
                    (self.box1_x_y_length[0] - self.box1_x_y_length[2]/2.0),
                    (self.box1_x_y_length[1] - self.box1_x_y_length[2]/2.0),
                    (self.box1_x_y_length[0] + self.box1_x_y_length[2]/2.0),
                    (self.box1_x_y_length[1] + self.box1_x_y_length[2]/2.0)
                    ])
        self.box2_x_y_length = np.array([-1.25, 2, 1.5])  # Bottom left.
        self.corners2 = np.array([
                    (self.box2_x_y_length[0] - self.box2_x_y_length[2]/2.0),
                    (self.box2_x_y_length[1] - self.box2_x_y_length[2]/2.0),
                    (self.box2_x_y_length[0] + self.box2_x_y_length[2]/2.0),
                    (self.box2_x_y_length[1] + self.box2_x_y_length[2]/2.0)
                    ])
        self.box3_x_y_length = np.array([0, 6, 1.5])  # Top middle.
        self.corners3 = np.array([
                    (self.box3_x_y_length[0] - self.box3_x_y_length[2]/2.0),
                    (self.box3_x_y_length[1] - self.box3_x_y_length[2]/2.0),
                    (self.box3_x_y_length[0] + self.box3_x_y_length[2]/2.0),
                    (self.box3_x_y_length[1] + self.box3_x_y_length[2]/2.0)
                    ])

        # Target set parameters.
        self.box4_x_y_length = np.array([0, 7+1.5, 1.5])  # Top.

        # Gym variables.
        self.action_space = gym.spaces.Discrete(3)  # horizontal_rate = {-1,0,1}
        self.midpoint = (self.low + self.high)/2.0
        self.interval = self.high - self.low
        self.observation_space = gym.spaces.Box(self.midpoint - self.interval,
                                                self.midpoint + self.interval)
        self.viewer = None

        # Discretization.
        self.grid_cells = None

        # Internal state.
        self.state = np.zeros(3)

        self.seed_val = 0

        # Visualization params
        self.vis_init_flag = True
        (self.x_box1_pos, self.x_box2_pos,
         self.x_box3_pos, self.y_box1_pos,
         self.y_box2_pos, self.y_box3_pos) = self.constraint_set_boundary()
        (self.x_box4_pos, self.y_box4_pos) = self.target_set_boundary()
        self.visual_initial_states = [np.array([0, 0]),
                                      np.array([-1, -1.9]),
                                      np.array([1, -1.9]),
                                      np.array([-1, 4]),
                                      np.array([1, 4])]
        self.scaling = 4.0

        # Set random seed.
        np.random.seed(self.seed_val)

    def reset(self, start=None):
        """ Reset the state of the environment.

        Args:
            start: Which state to reset the environment to. If None, pick the
                state uniformly at random.

        Returns:
            The state the environment has been reset to.
        """
        if start is None:
            self.state = self.sample_random_state()
        else:
            self.state = start
        return np.copy(self.state)

    def sample_random_state(self):
        # Sample states uniformly until one is found inside the constraint set
        # but outside target.
        # rnd_sector = np.random.randint(0, 3)
        # if rnd_sector == 0:
        #     rnd_state = np.random.uniform(low=self.low,
        #                                   high=np.array([self.corners1[2],
        #                                                  self.corners1[1]]))
        # elif rnd_sector == 1:
        #     rnd_state = np.random.uniform(low=np.array([self.corners2[0],
        #                                                 self.corners2[3]]),
        #                                   high=np.array([self.corners3[0],
        #                                                  self.high[1]]))
        # elif rnd_sector == 2:
        #     rnd_state = np.random.uniform(low=np.array([self.corners3[2],
        #                                                 self.corners1[3]]),
        #                                   high=self.high)
        # else:
        #     print("ERROR rnd_sector needs to be 0, 1 or 2.")
        rnd_state = np.random.uniform(low=self.low,
                                      high=self.high)
        # while ((self.safety_margin(rnd_state) > 0) or (
        #         self.target_margin(rnd_state) > 0)):
        #     rnd_state = np.random.uniform(low=self.low, high=self.high)
        return rnd_state

    def step(self, action):
        """ Evolve the environment one step forward under given input action.

        Args:
            action: Input action.

        Returns:
            Tuple of (next state, signed distance of current state, whether the
            episode is done, info dictionary).
        """
        # The signed distance must be computed before the environment steps
        # forward.
        if self.grid_cells is None:
            l_x = self.target_margin(self.state)
            g_x = self.safety_margin(self.state)
        else:
            nearest_point = nearest_real_grid_point(
                self.grid_cells, self.bounds, self.state)
            l_x = self.target_margin(nearest_point)
            g_x = self.safety_margin(nearest_point)

        # Move dynamics one step forward.
        x, y = self.state
        u = self.discrete_controls[action]

        x, y = self.integrate_forward(x, y, u)
        self.state = np.array([x, y])

        # Calculate whether episode is done.
        done = ((g_x > 0) or (l_x <= 0))
        info = {"g_x": g_x}
        return np.copy(self.state), l_x, done, info

    def integrate_forward(self, x, y, u):
        """ Integrate the dynamics forward by one step.

        Args:
            x: Position in x-axis.
            y: Position in y-axis
            theta: Heading.
            u: Contol input.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        x = x + self.time_step * u
        y = y + self.time_step * self.upward_speed
        return x, y

    def set_seed(self, seed):
        """ Set the random seed.

        Args:
            seed: Random seed.
        """
        self.seed_val = seed
        np.random.seed(self.seed_val)

    def safety_margin(self, s):
        """ Computes the margin (e.g. distance) between state and failue set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        box1_safety_margin = -(np.linalg.norm(s - self.box1_x_y_length[:2],
                               ord=np.inf) - self.box1_x_y_length[-1]/2.0)
        box2_safety_margin = -(np.linalg.norm(s - self.box2_x_y_length[:2],
                               ord=np.inf) - self.box2_x_y_length[-1]/2.0)
        box3_safety_margin = -(np.linalg.norm(s - self.box3_x_y_length[:2],
                               ord=np.inf) - self.box3_x_y_length[-1]/2.0)

        vertical_margin = (np.abs(s[1] - (self.low[1] + self.high[1])/2.0)
                           - self.interval[1]/2.0)
        horizontal_margin = np.abs(s[0]) - 2.0
        enclosure_safety_margin = max(horizontal_margin, vertical_margin)

        safety_margin = max(box1_safety_margin,
                            box2_safety_margin,
                            box3_safety_margin,
                            enclosure_safety_margin)

        return self.scaling * safety_margin

    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        box4_target_margin = (np.linalg.norm(s - self.box4_x_y_length[:2],
                              ord=np.inf) - self.box4_x_y_length[-1]/2.0)

        target_margin = box4_target_margin
        return self.scaling * target_margin

    def set_grid_cells(self, grid_cells):
        """ Set number of grid cells.

        Args:
            grid_cells: Number of grid cells as a tuple.
        """
        self.grid_cells = grid_cells

        # (self.x_opos, self.y_opos, self.x_ipos,
        #  self.y_ipos) = self.constraint_set_boundary()

    def set_bounds(self, bounds):
        """ Set state bounds.

        Args:
            bounds: Bounds for the state.
        """
        self.bounds = bounds

        # Get lower and upper bounds
        self.low = np.array(self.bounds)[:, 0]
        self.high = np.array(self.bounds)[:, 1]

        # Double the range in each state dimension for Gym interface.
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(midpoint - interval,
                                                midpoint + interval)

    def render(self, mode='human'):
        pass

    # def ground_truth_comparison(self, q_func):
    #     """ Compares the state-action value function to the ground truth.

    #     The state-action value function is first converted to a state value
    #     function, and then compared to the ground truth analytical solution.

    #     Args:
    #         q_func: State-action value function.

    #     Returns:
    #         Tuple containing number of misclassified safe and misclassified
    #         unsafe states.
    #     """
    #     computed_v = v_from_q(
    #         q_values_from_q_func(q_func, self.grid_cells, self.bounds, 2))
    #     return self.ground_truth_comparison_v(computed_v)

    # def ground_truth_comparison_v(self, computed_v):
    #     """ Compares the state value function to the analytical solution.

    #     The state value function is compared to the ground truth analytical
    #     solution by checking for sign mismatches between state-value pairs.

    #     Args:
    #         computed_v: State value function.

    #     Returns:
    #         Tuple containing number of misclassified safe and misclassified
    #         unsafe states.
    #     """
    #     analytic_v = self.analytic_v()
    #     misclassified_safe = 0
    #     misclassified_unsafe = 0
    #     it = np.nditer(analytic_v, flags=['multi_index'])
    #     while not it.finished:
    #         if analytic_v[it.multi_index] < 0 < computed_v[it.multi_index]:
    #             misclassified_safe += 1
    #         elif computed_v[it.multi_index] < 0 < analytic_v[it.multi_index]:
    #             misclassified_unsafe += 1
    #         it.iternext()
    #     return misclassified_safe, misclassified_unsafe

    def constraint_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        x_box1_pos = np.array([
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[0] - self.box1_x_y_length[-1]/2.0])
        x_box2_pos = np.array([
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[0] - self.box2_x_y_length[-1]/2.0])
        x_box3_pos = np.array([
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[0] - self.box3_x_y_length[-1]/2.0])

        y_box1_pos = np.array([
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] + self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0,
            self.box1_x_y_length[1] - self.box1_x_y_length[-1]/2.0])
        y_box2_pos = np.array([
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] + self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0,
            self.box2_x_y_length[1] - self.box2_x_y_length[-1]/2.0])
        y_box3_pos = np.array([
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] + self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0,
            self.box3_x_y_length[1] - self.box3_x_y_length[-1]/2.0])

        return (x_box1_pos, x_box2_pos, x_box3_pos,
                y_box1_pos, y_box2_pos, y_box3_pos)

    def target_set_boundary(self):
        """ Computes the safe set boundary based on the analytic solution.

        The boundary of the safe set for the double integrator is determined by
        two parabolas and two line segments.

        Returns:
            Set of discrete points describing each parabola. The first and last
            two elements of the list describe the set of coordinates for the
            first and second parabola respectively.
        """
        x_box4_pos = np.array([
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[0] - self.box4_x_y_length[-1]/2.0])

        y_box4_pos = np.array([
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] + self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0,
            self.box4_x_y_length[1] - self.box4_x_y_length[-1]/2.0])

        return (x_box4_pos, y_box4_pos)

    def visualize_analytic_comparison(self, v, no_show=False,
                                      labels=["x dot", "x"]):
        """ Overlays analytic safe set on top of state value function.

        Args:
            v: State value function.
        """
        plt.clf()
        # boundary = ((v < 0.1) * (v > -0.1))
        # v[boundary] = np.max(v)
        visualize_matrix(v.T, self.get_axes(labels), no_show)

        # Plot bounadries of constraint set.
        plt.plot(self.x_box1_pos, self.y_box1_pos, color="black")
        plt.plot(self.x_box2_pos, self.y_box2_pos, color="black")
        plt.plot(self.x_box3_pos, self.y_box3_pos, color="black")
        # Plot boundaries of target set.
        plt.plot(self.x_box4_pos, self.y_box4_pos, color="black")

    def simulate_one_trajectory(self, q_func, T=10, state=None):

        if state is None:
            state = self.sample_random_state()
        x, y = state
        traj_x = [x]
        traj_y = [y]

        for t in range(T):
            if self.safety_margin(state) > 0 or self.target_margin(state) < 0:
                break
            state_ix = state_to_index(self.grid_cells, self.bounds, state)
            action_ix = np.argmin(q_func[state_ix])
            u = self.discrete_controls[action_ix]

            x, y = self.integrate_forward(x, y, u)
            state = np.array([x, y])
            traj_x.append(x)
            traj_y.append(y)

        return traj_x, traj_y

    def simulate_trajectories(self, q_func, T=10, num_rnd_traj=None,
                              states=None):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = []

        if states is None:
            for _ in range(num_rnd_traj):
                trajectories.append(self.simulate_one_trajectory(q_func, T=T))
        else:
            for state in states:
                trajectories.append(
                    self.simulate_one_trajectory(q_func, T=T, state=state))

        return trajectories

    def plot_trajectories(self, q_func, T=10, num_rnd_traj=None, states=None):

        assert ((num_rnd_traj is None and states is not None) or
                (num_rnd_traj is not None and states is None) or
                (len(states) == num_rnd_traj))
        trajectories = self.simulate_trajectories(q_func, T=T,
                                                  num_rnd_traj=num_rnd_traj,
                                                  states=states)

        for traj in trajectories:
            traj_x, traj_y = traj
            plt.plot(traj_x, traj_y, color="black")

    # def analytic_v(self):
    #     """ Computes the discretized analytic value function.

    #     Returns:
    #         Discretized form of the analytic state value function.
    #     """
    #     x_low = self.target_low[0]
    #     x_high = self.target_high[0]
    #     u_max = self.control_bounds[1]  # Assumes u_max = -u_min.

    #     def analytic_function(x, x_dot):
    #         if x_dot >= 0:
    #             return min(x - x_low,
    #                        x_high - x - x_dot ** 2 / (2 * u_max))
    #         else:
    #             return min(x_high - x,
    #                        x - x_dot ** 2 / (2 * u_max) - x_low)

    #     v = np.zeros(self.grid_cells)
    #     it = np.nditer(v, flags=['multi_index'])
    #     while not it.finished:
    #         x, x_dot = index_to_state(self.grid_cells, self.bounds,
    #                                   it.multi_index)
    #         v[it.multi_index] = analytic_function(x, x_dot)
    #         it.iternext()
    #     return v

    def get_axes(self, labels=["x", "y"]):
        """ Gets the bounds for the environment.

        Returns:
            List containing a list of bounds for each state coordinate and a
            list for the name of each state coordinate.
        """
        return [np.append(self.bounds[0], self.bounds[1]), labels]
