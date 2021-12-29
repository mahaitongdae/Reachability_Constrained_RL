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

matplotlib.use("TkAgg")
matplotlib.style.use('ggplot')


class DubinsCarEnv(gym.Env):

    def __init__(self):

        # State bounds.
        self.bounds = np.array([[-1.1, 1.1],  # axis_0 = state, axis_1 = bounds.
                                [-1.1, 1.1],
                                [0, 2*np.pi]])
        self.low = self.bounds[:, 0]
        self.high = self.bounds[:, 1]

        # Time step parameter.
        self.time_step = 0.05

        # Dubins car parameters.
        self.speed = 1.0

        # Control parameters.
        # TODO{vrubies: Check proper rates.}
        self.max_turning_rate = 1.5
        self.discrete_controls = np.array([-self.max_turning_rate,
                                           self.max_turning_rate])

        # Constraint set parameters.
        self.inner_radius = 0.25
        self.outer_radius = 1.0

        # Target set parameters.
        self.target_radius = self.inner_radius
        # self.target_radius = 1.0/16.0
        self.target_center_x = (self.inner_radius + self.outer_radius) / 2.0
        self.target_center_y = 0
        # self.target_center = np.array([self.target_center_x,
        #                                self.target_center_y])
        self.target_center = np.array([0,
                                       0])

        # Gym variables.
        self.action_space = gym.spaces.Discrete(2)  # angular_rate = {-1,1}
        midpoint = (self.low + self.high)/2.0
        interval = self.high - self.low
        self.observation_space = gym.spaces.Box(midpoint - interval,
                                                midpoint + interval)
        self.viewer = None

        # Discretization.
        self.grid_cells = None

        # Internal state.
        self.state = np.zeros(3)

        self.seed_val = 0

        # Visualization params
        self.angle_slices = [0]
        self.vis_init_flag = True
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
            x_rnd, y_rnd, theta_rnd = self.sample_random_state()
            self.state = np.array([x_rnd, y_rnd, theta_rnd])
        else:
            self.state = start
        return np.copy(self.state)

    def sample_random_state(self):
        # Sample between -pi to pi.
        angle = (2.0 * np.random.uniform() - 1.0) * np.pi

        # Sample inside a ring uniformly at random.
        dist = np.sqrt(np.random.uniform() *
                       (self.outer_radius**2 - self.inner_radius**2) +
                       self.inner_radius**2)
        assert (dist <= self.outer_radius and dist >= self.inner_radius)
        x_rnd = dist * np.cos(angle)
        y_rnd = dist * np.sin(angle)
        theta_rnd = np.random.uniform(low=self.low[-1], high=self.high[-1])
        return x_rnd, y_rnd, theta_rnd

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
        x, y, theta = self.state
        u = self.discrete_controls[action]

        x, y, theta = self.integrate_forward(x, y, theta, u)
        self.state = np.array([x, y, theta])

        # Calculate whether episode is done.
        dist_origin = np.linalg.norm(self.state[:2])
        done = ((g_x > 0.0) or (l_x <= 0.0))  # TODO(vrubies) more efficient in 142.
        info = {"g_x": g_x}
        return np.copy(self.state), l_x, done, info

    def integrate_forward(self, x, y, theta, u):
        """ Integrate the dynamics forward by one step.

        Args:
            x: Position in x-axis.
            y: Position in y-axis
            theta: Heading.
            u: Contol input.

        Returns:
            State variables (x,y,theta) integrated one step forward in time.
        """
        x = x + self.time_step * self.speed * np.cos(theta)
        y = y + self.time_step * self.speed * np.sin(theta)
        theta = np.mod(theta + self.time_step * u, 2*np.pi)
        return x, y, theta

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
        dist_to_origin = np.linalg.norm(s[:2])
        outer_dist = dist_to_origin - self.outer_radius
        # inner_dist = self.inner_radius - dist_to_origin
        # Note the "-" sign. This ensures x \in K \iff g(x) <= 0.
        # safety_margin = maxmin?(outer_dist, inner_dist)
        safety_margin = outer_dist
        # if safety_margin > 0:
        #     safety_margin = 10
        # if x_in:
        #     return -1 * x_dist
        return self.scaling * safety_margin

    def target_margin(self, s):
        """ Computes the margin (e.g. distance) between state and target set.

        Args:
            s: State.

        Returns:
            Margin for the state s.
        """
        dist_to_target = np.linalg.norm(s[:2] - self.target_center)
        target_margin = dist_to_target - self.target_radius
        # if x_in:
        #     return -1 * x_dist
        return self.scaling * target_margin

    def set_grid_cells(self, grid_cells):
        """ Set number of grid cells.

        Args:
            grid_cells: Number of grid cells as a tuple.
        """
        self.grid_cells = grid_cells

        (self.x_opos, self.y_opos, self.x_ipos,
         self.y_ipos) = self.constraint_set_boundary()

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
        num_points = self.grid_cells[0]
        x_opos = np.zeros((num_points,))
        y_opos = np.zeros((num_points,))
        x_ipos = np.zeros((num_points,))
        y_ipos = np.zeros((num_points,))
        linspace = 2.1 * np.pi * np.arange(start=0, stop=1, step=1/num_points)

        x_opos = self.outer_radius * np.cos(linspace)
        y_opos = self.outer_radius * np.sin(linspace)
        x_ipos = self.inner_radius * np.cos(linspace)
        y_ipos = self.inner_radius * np.sin(linspace)

        return (x_opos, y_opos, x_ipos, y_ipos)

    def visualize_analytic_comparison(self, v, no_show=False,
                                      labels=["x", "y"]):
        """ Overlays analytic safe set on top of state value function.

        Args:
            v: State value function.
        """
        plt.clf()
        # Plot analytic parabolas.
        plt.plot(self.x_opos, self.y_opos, color="black")
        plt.plot(self.x_ipos, self.y_ipos, color="black")

        # num_subfigs = len(self.angle_slices)
        # if self.vis_init_flag:
        #     fig, ax = plt.subplots(nrows=1, ncols=num_subfigs)
        #     self.vis_init_flag = False
        # for ii in range(num_subfigs):
        #     plt.subplot(1, num_subfigs, ii+1)
        #     # Visualize state value.
        visualize_matrix(v[:, :, self.angle_slices[0]].T,
                         self.get_axes(labels), no_show)

    def simulate_one_trajectory(self, q_func, T=10, state=None):

        if state is None:
            state = self.sample_random_state()
        x, y, theta = state
        traj_x = [x]
        traj_y = [y]

        for t in range(T):
            if self.safety_margin(state) > 0 or self.target_margin(state) < 0:
                break
            state_ix = state_to_index(self.grid_cells, self.bounds, state)
            action_ix = np.argmin(q_func[state_ix])
            u = self.discrete_controls[action_ix]

            x, y, theta = self.integrate_forward(x, y, theta, u)
            state = np.array([x, y, theta])
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
