# Copyright (c) 2019–2020, The Regents of the University of California.
# All rights reserved.
#
# This file is subject to the terms and conditions defined in the LICENSE file
# included in this repository.
#
# Please contact the author(s) of this library if you have any questions.
# Authors: Neil Lugovoy   ( nflugovoy@berkeley.edu )

import _pickle as cPickle
import os
from datetime import datetime
import itertools
from glob import glob

import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import tensorflow as tf

# == backup and outcome functions ==


def sbe_backup(rewards, dones, next_state_val, gamma, tensorflow=False):
    """
    computes backup for Safety Bellman Equation from equation 7
    :param rewards: either a scalar reward or a tensor of rewards for multiple trajectories
    :param dones: either a single boolean representing whether the state is terminal or a tensor of
    dones for multiple trajectories
    :param next_state_val: the value function at the next state (e.g. in q-learning this the max
    over actions of the q values in the next state) or a tensor of such values
    :param gamma: discount factor
    :param tensorflow: whether the tensor is a tensorflow tensor
    :return: the value for the backup
    """
    v_terminal = rewards
    if tensorflow:
        v_non_terminal = (1.0 - gamma) * rewards + gamma * tf.minimum(rewards, next_state_val)
    else:
        v_non_terminal = (1.0 - gamma) * rewards + gamma * np.minimum(rewards, next_state_val)
    return dones * v_terminal + (1.0 - dones) * v_non_terminal


def sbe_outcome(rewards, gamma):
    """
    computes the outcome of the trajectory with the given discount factor for the Safety Bellman
    Equation from equation 8. If you want to use this with a value function predicting the value at
    the last state append that value to the end of rewards. See the rewards_plus_v variable in ray's
    rllib/evaluation/postprocessing.py file for an example.
    :param rewards: 1d list or array of rewards for the trajectory
    :param gamma: discount factor
    :return: a list such that the value at index i is the outcome starting from the ith state in
    the trajectory. at i=0 is the outcome for the entire trajectory
    """
    outcomes = np.zeros(len(rewards))
    outcomes[-1] = rewards[-1]
    for i in range(len(outcomes) - 2, -1, -1):
        outcomes[i] = (1 - gamma) * rewards[i] + gamma * min(rewards[i], outcomes[i+1])
    return outcomes

# == discretization functions ==


def state_to_index(grid_cells, state_bounds, state):
    """

    :param grid_cells: tuple of ints where the ith value is the number of grid_cells for ith dimension of
    state
    :param state_bounds: list of tuples where ith tuple contains the min and max value in that order
     of ith dimension
    :param state: state to discretize
    :return: state discretized into appropriate grid_cells
    """
    index = []
    for i in range(len(state)):
        lower_bound = state_bounds[i][0]
        upper_bound = state_bounds[i][1]
        if state[i] <= lower_bound:
            index.append(0)
        elif state[i] >= upper_bound:
            index.append(grid_cells[i] - 1)
        else:
            index.append(int(
                ((state[i] - lower_bound) * grid_cells[i]) //
                (upper_bound - lower_bound)))
    return tuple(index)


def index_to_state(grid_cells, state_bounds, discrete):
    """

    :param grid_cells: tuple of ints where the ith value is the number of grid_cells for ith dimension of
    state
    :param state_bounds: list of tuples where ith tuple contains the min and max value in that order
     of ith dimension
    :param discrete: discrete state to approximate to nearest real value
    :return: the real valued state at the center of the grid_cells of the discrete state, an "inverse"
    of state_to_index
    """
    state = np.zeros(len(discrete))
    for i in range(len(discrete)):
        if grid_cells[i] == 1:
            state[i] = ((state_bounds[i][1] - state_bounds[i][0]) / 2) + state_bounds[i][0]
        else:
            scaling = (grid_cells[i] - 1) / (state_bounds[i][1] - state_bounds[i][0])
            state[i] = discrete[i] / scaling + state_bounds[i][0]
    return state


def nearest_real_grid_point(grid_cells, state_bounds, state):
    """

    :param grid_cells: tuple of ints where the ith value is the number of grid_cells for ith dimension of
     state
    :param state_bounds: list of tuples where ith tuple contains the min and max value in that order
    of ith dimension
    :param state: state to convert to center of bucket
    :return: the real valued state at the center of the bucket that state would go into
    """
    return index_to_state(grid_cells, state_bounds, state_to_index(grid_cells,
                                                                   state_bounds,
                                                                   state))

# == q value and value function utils ==


def v_from_q(q_values):
    """
    Compute the state value function from the state-action value function by taking the maximum
    over available actions at each state.
    :param q_values: q value function tensor numpy array
    :return: value function numpy array
    """
    v = np.zeros(np.shape(q_values)[:-1])
    it = np.nditer(q_values, flags=['multi_index', 'refs_ok'])
    while not it.finished:
        v[it.multi_index[:-1]] = min(q_values[it.multi_index[:-1]])
        it.iternext()
    return v


def q_values_from_q_func(q_func, num_grid_cells, state_bounds, action_n):
    """
    computes q value tensor from a q value function
    :param q_func: function from state to q value
    :param num_grid_cells: number of grid_cells for resulting q value tensor
    :param state_bounds: state bounds for resulting q value tensor
    :param action_n: number of actions in action space
    :return: q value tensor as numpy array
    """
    q_values = np.zeros(num_grid_cells + (action_n,))
    it = np.nditer(q_values, flags=['multi_index'])
    while not it.finished:
        qs = q_func(index_to_state(num_grid_cells, state_bounds=state_bounds,
                                     discrete=it.multi_index[:-1]))
        q_values[it.multi_index] = qs[0]
        it.iternext()
    return q_values


# == plotting functions ==


def visualize_matrix(m, axes=None, no_show=False):
    if axes is not None:
        # Transpose is necessary so that m[x,y] is (x,y) on plot.
        f = plt.imshow(m, interpolation='none', extent=axes[0], origin="lower",
                       cmap="plasma", vmin=-2, vmax=4)
        a = plt.gca()
        a.set_aspect((axes[0][1]-axes[0][0])/(axes[0][3]-axes[0][2]))  # makes equal aspect ratio
        a.set_xlabel(axes[1][0])
        a.set_ylabel(axes[1][1])
        a.grid(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
        )
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
        )
        a.set_xticklabels([])
        a.set_yticklabels([])
    else:
        f = plt.imshow(m, interpolation='none', origin="lower", cmap="plasma")

    if not no_show:
        plt.show()
    return f


def visualize_state_visits(state_visits):
    state_visits = state_visits.reshape(np.prod(np.shape(state_visits)))
    plt.bar(range(len(state_visits)), state_visits)
    plt.xlabel("state")
    plt.ylabel("number of visits")
    axes = plt.gca()
    axes.set_ylim([0, 150000])
    plt.tight_layout()
    plt.show()

# == schedules ==


class SteppedSchedule(object):
    def __init__(self, initial_value, final_value, half_life):
        """Stepped exponential schedule to anneal gamma towards 1
        Parameters
        ----------
        initial_value: float
            the starting value
          to final_p
        final_value: float
            final value that the schedule will stop increasing at
        half_life: int
            number of timesteps that the value stays constant before increasing
        """
        self.half_life = half_life
        self.final_value = final_value
        self.initial_value = initial_value

    def value(self, t):
        """See Schedule.value"""
        c = 1 - self.initial_value
        return min(self.final_value, 1 - c * (2 ** -(t // self.half_life)))


def make_stepped_schedule(start_value, half_life, max_gamma):
    c = 1 - start_value
    return lambda t, n: np.minimum(max_gamma, 1 - c * 2 ** (-t // half_life))


def make_linear_schedule(start_value, end_value, decay_steps):
    m = (end_value-start_value)/decay_steps
    b = start_value
    return lambda t: max(m * t + b, end_value)


def make_stepped_linear_schedule(start_value, end_value, total_time,
                                 steps=1):
    width = total_time/steps
    m = (end_value-start_value)/total_time
    b = start_value
    stair_values = np.array([m * (i * width) + b for i in range(steps+1)])
    width2 = total_time/(steps+1)
    times_values = np.array([(i * width2) for i in range(steps+1)])
    return lambda t, n: stair_values[times_values <= t][-1]


def make_log_decay_schedule(initial, decay):
    return lambda t, n: initial / (1 + decay * np.log(t + 1))


def make_inverse_polynomial_visit_schedule(scale, power):
    return lambda t, n: scale * 1 / (n ** power)


def make_inverse_visit_schedule(episodic_length):
    return lambda t, n: (episodic_length + 1) / (episodic_length + n)

# == data collection functions ==

def compare_against_rollout(horizon, n_samples, q_func, env):
    """
    compares the predicted value of the q_func at the start of the trajectory to the true minimum
    achieved after acting on-policy acording to the q_func
    :param horizon: maximum trajectory length
    :param n_samples: number of trajectories to sample
    :param q_func: a function that takes in the state and outputs the q values for each action at
    that state
    :param env: environment to simulate in
    :return: a n_samples long list of tuples of (actual, predicted)
    """
    rollout_comparisons = []

    for i in range(n_samples):
        s = env.reset()
        q_values = q_func(s)
        predicted = np.max(q_values)
        actual = float("inf")
        done = False
        for t in itertools.count():
            if done or t >= horizon:
                break
            s, r, done, info = env.step(np.argmax(q_values))
            actual = min(actual, r)
            q_values = q_func(s)
        rollout_comparisons.append((float(actual), float(predicted)))
    return rollout_comparisons


def eval_violation(time_horizon, n_samples, q_values, env):
    """
    counts how many times the policy violates the safety constraints on n_samples many trajectories
    of length time_horizon. Acts on-policy by taking action to be argmax of q_values and uses the
    env to simulate. Since the starting state is sampled uniformly at random this can be used as an
    unbiased estimator for the fraction of states that are in the safe set for the provided policy
    and time horizon
    :param time_horizon: maximum trajectory length
    :param n_samples: number of trajectories to sample for
    :param q_values: a function that takes in the state and outputs the q values for each action at
    that state
    :param env: environment to simulate in
    :return: number of episodes that had a violation
    """
    violations = 0
    for _ in range(n_samples):
        s = env.reset()
        for t in range(time_horizon):
            s, r, done, info = env.step(np.argmax(q_values(s)))
            if r <= 0:
                violations += 1
                break
            if done:
                break
    return violations


def offsets(d):
    """
    this function calculates all the possible index offsets for cells next to a cell in a d
    dimensional grid. for example d = 1 we have [[-1],[0],[1]] and for d = 2 we have
    [[-1, -1],[-1,  0],[-1,  1],[ 0, -1],[0, 0],[0, 1],[ 1, -1],[1, 0],[1, 1]]. this is used for
    checking the adjacent grid cells when comparing value functions
    :param d: the dimension of the grid
    """
    l = []
    for bit_string in ["".join(seq) for seq in itertools.product("012", repeat=d)]:
        l.append(np.array([int(b) - 1 for b in bit_string]))
    return l

# == save directory ==


def get_save_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


# == saving and loading q functions == # TODO this still needs some cleanup

def save(q_values, stats, experiment_name, save_dir=None):
    """
    saves q_values and stats to a directory named experiment name concatenated with the date
    inside of save_dir. The file is named time_iteration.pickle. The date, time and iteration are
    taken from the stats dictionary. The date and time are the date and time the experiment started.
    :param q_values: Q value function to be saved
    :param stats: stats to be saved
    :param experiment_name: name of experiment. files will be saved to a directory named
    experiment_name_date
    :param save_dir: the parent directory of the experiment directory. If left as None will use
    get_save_dir()
    :return: the directory saved in
    """
    if save_dir is None:
        save_dir = get_save_dir()

    # set up directory and file names
    date, time = stats['start_time'].split()
    directory = os.path.join(save_dir, experiment_name + '_' + date)
    file_name = time + '_' + str(stats['episode']) + '.pickle'
    path = os.path.join(directory, file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # set up dictionary to save data
    dictionary = {"q_values": q_values, "stats": stats}

    try:
        with open(path, 'wb') as handle:
            cPickle.dump(dictionary, handle)
    except Exception as e:
        print("Error saving q_values and stats: \n", e)
    return path


def load(path):
    """

    :param directory: directory to load from
    :return: Q and stats loaded from directory
    """
    try:
        with open(path, 'rb') as handle:
            dictionary = cPickle.load(handle)
        q_values = dictionary["q_values"]
        stats = dictionary["stats"]
        return q_values, stats
    except Exception as e:
        print("Error loading: \n", e)


def load_most_recent(directory):
    files = glob(directory)
    most_recent = None
    most_recent_time = None
    most_recent_iteration = None
    for file in files:
        file = os.path.splitext(file)[0]
        time, iteration = file.split("_")
        time = datetime.strptime(time, '%H:%M:%S')
        if most_recent is None or (time > most_recent_time and iteration > most_recent_iteration):
            most_recent = file
            most_recent_time = time
            most_recent_iteration = iteration
    return load(os.path.join(directory, most_recent))
