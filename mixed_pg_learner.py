import numpy as np
import gym
from collections import OrderedDict

from gym.envs.user_defined.path_tracking_env import EnvironmentModel
from preprocessor import Preprocessor
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TimerStat:
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    @property
    def mean(self):
        return np.mean(self._samples)


class MixedPGLearner(object):
    import tensorflow as tf

    def __init__(self, policy_cls, args):
        self.args = args
        env = gym.make(self.args.env_id)
        obs_space, act_space = env.observation_space, env.action_space
        self.path = env.path
        env.close()
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.policy_for_rollout = policy_cls(obs_space, act_space, self.args)
        self.batch_data = {}
        self.M = 1
        self.num_rollout_list = [0, 20, 40, 60]
        self.model = EnvironmentModel()
        self.preprocessor = Preprocessor(obs_space, self.args.obs_normalize, self.args.reward_preprocess_type,
                                         self.args.reward_scale_factor, gamma=self.args.gamma)
        self.forward_timer = TimerStat()
        self.backward_timer = TimerStat()

    def get_batch_data(self, batch_data):
        self.batch_data = self.post_processing(batch_data)
        batch_advs, batch_tdlambda_returns = self.compute_advantage()
        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns))
        self.shuffle()

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               'batch_neglogps': np.asarray(list(map(lambda x: x[5], batch_data)), dtype=np.float32)}
        return tmp

    def batch_data_count(self):
        return len(self.batch_data['batch_obs'])

    def shuffle(self):
        permutation = np.random.permutation(self.batch_data_count())
        for key, val in self.batch_data.items():
            self.batch_data[key] = val[permutation]

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])
        processed_batch_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()
        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()

        batch_values = self.policy_with_value.compute_Qs(processed_batch_obs[:, :-1],
                                                         self.batch_data['batch_actions'])[0].numpy()[:,
                       0]  # len = n_steps
        batch_advs = np.zeros_like(self.batch_data['batch_rewards'], dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t + 1]
            delta = processed_batch_rewards[t] + self.args.gamma * batch_values[t + 1] * nextnonterminal - \
                    batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values
        return batch_advs, batch_tdlambda_returns

    def model_rollout_return(self, num_traj, num_rollout_list, start_obses, start_actions):
        max_num_rollout = max(num_rollout_list)
        n = len(num_rollout_list)

        self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
        obses_tile = self.tf.tile(start_obses, [num_traj, 1])
        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        actions_tile = self.tf.tile(start_actions, [num_traj, 1])
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0], 1))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0], 1))]

        self.model.reset(obses_tile)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                print(ri)
                obses_tile, rewards = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * self.tf.reshape(processed_rewards, (-1, 1))
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_for_rollout.compute_action(processed_obses_tile[:, :-1])
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri+1) * self.tf.ones((obses_tile.shape[0], 1)))

        all_Qs = self.policy_for_rollout.compute_Qs(
            self.tf.concat(processed_obses_tile_list, 0)[:, :-1], self.tf.concat(actions_tile_list, 0))[0]
        all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
        all_gammas = self.tf.concat(gammas_list, 0)

        final = self.tf.reshape(all_rewards_sums + all_gammas * all_Qs, (max_num_rollout+1, num_traj, -1))
        # final [[[time0+traj0], [time0+traj1], ..., [time0+trajn]],
        #        [[time1+traj0], [time1+traj1], ..., [time1+trajn]],
        #        ...
        #        [[timen+traj0], [timen+traj1], ..., [timen+trajn]],
        #        ]
        model_returns = self.tf.reduce_mean(final, axis=1)
        out_returns = []
        for num_rollout in num_rollout_list:
            out_returns.append(model_returns[num_rollout])

        out = self.tf.concat(out_returns, 0)

        return out

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        processed_mb_obs = self.preprocessor.np_process_obses(mb_obs)
        mb_advs = self.batch_data['batch_advs'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        mb_neglogps = self.batch_data['batch_neglogps'][start_idx: end_idx]

        with self.tf.GradientTape() as tape:
            q_pred = self.policy_with_value.compute_Qs(processed_mb_obs[:, :-1], mb_actions)[0]
            q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - mb_tdlambda_returns))
        q_gradient = tape.gradient(q_loss,
                                   self.policy_with_value.models[0].trainable_weights,
                                   unconnected_gradients=self.tf.UnconnectedGradients.ZERO)  # TODO

        policy_gradient_list = []
        heuristic_bias_list = []
        var_list = []

        with self.forward_timer:
            with self.tf.GradientTape() as tape:
                acts, _ = self.policy_with_value.compute_action(processed_mb_obs[:, :-1])
                model_returns = self.model_rollout_return(self.M, self.num_rollout_list, mb_obs, acts)
        logger.info('forward time: {}'.format(self.forward_timer.mean))
        with self.backward_timer:
            print(11111)
            jaco = tape.jacobian(model_returns,
                                 self.policy_with_value.models[1].trainable_weights,
                                 unconnected_gradients=self.tf.UnconnectedGradients.ZERO,
                                 experimental_use_pfor=True)  # TODO
        # [[dy1/dx1, dy2/dx1,...(rolloutnum1)|dy1/dx1, dy2/dx1,...(rolloutnum2)| ...],
        #  [dy1/dx2, dy2/dx2, ...(rolloutnum1)|dy1/dx2, dy2/dx2,...(rolloutnum2)| ...],
        #  ...]
        logger.info('backward time: {}'.format(self.backward_timer.mean))

        for rollout_index in range(len(self.num_rollout_list)):
            jaco_for_this_rollout = list(map(lambda x: x[rollout_index*self.args.mini_batch_size:
                                                         (rollout_index+1)*self.args.mini_batch_size], jaco))

            gradient_std = []
            gradient_mean = []
            var = 0
            for x in jaco_for_this_rollout:
                gradient_std.append(self.tf.math.reduce_std(x, 0))
                gradient_mean.append(self.tf.reduce_mean(x, 0))
                var += self.tf.reduce_mean(self.tf.square(gradient_std[-1]))
            heuristic_bias = self.tf.reduce_mean(self.tf.square(model_returns[rollout_index*self.args.mini_batch_size:
                                                         (rollout_index+1)*self.args.mini_batch_size]
                                                      - mb_tdlambda_returns))

            policy_gradient_list.append(gradient_mean)
            heuristic_bias_list.append(heuristic_bias)
            var_list.append(var)

        w_heur_bias_list = []
        w_var_list = []
        heuristic_bias_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1 / x, heuristic_bias_list)))
        var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1 / x, var_list)))
        for i in range(len(self.num_rollout_list)):
            w_heur_bias_list.append((1 / heuristic_bias_list[i]) / heuristic_bias_inverse_sum)
            w_var_list.append((1 / var_list[i]) / var_inverse_sum)

        w_list = []
        w_heur_bias_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1 / x, w_heur_bias_list)))
        w_var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1 / x, w_var_list)))
        for i in range(len(self.num_rollout_list)):
            w_list.append((1 / w_heur_bias_list[i] + 1 / w_var_list[i]) / (w_heur_bias_inverse_sum + w_var_inverse_sum))

        final_policy_gradient = []
        for i in range(len(policy_gradient_list[0])):
            tmp = 0
            for j in range(len(policy_gradient_list)):
                tmp += w_list[j] * policy_gradient_list[j][i]
            final_policy_gradient.append(tmp)

        gradient_tensor = q_gradient + final_policy_gradient
        return np.array(list(map(lambda x: x.numpy(), gradient_tensor)))


def test_vehicle_dynamics():
    vehicle_dynamics = VehicleDynamics()
    state = tf.Variable([[0., 0., 10., 0., 0., 0.]])
    action = tf.Variable([[0., 2.]])
    # state_deriv = vehicle_dynamics.f_xu(state, action)
    next_state = vehicle_dynamics.next_states(state, action)
    next_model_state = vehicle_dynamics.model_next_states(state, action)
    print(next_state, next_model_state)


if __name__ == '__main__':
    test_vehicle_dynamics()
