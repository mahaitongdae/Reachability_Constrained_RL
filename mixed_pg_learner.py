import numpy as np
import gym
from gym.envs.user_defined.path_tracking_env import EnvironmentModel
from preprocessor import Preprocessor
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# logger.setLevel(logging.INFO)

def judge_is_nan(list_of_np_or_tensor):
    for m in list_of_np_or_tensor:
        if hasattr(m, 'numpy'):
            if np.any(np.isnan(m.numpy())):
                print(list_of_np_or_tensor)
                raise ValueError
        else:
            if np.any(np.isnan(m)):
                print(list_of_np_or_tensor)
                raise ValueError


def judge_less_than(list_of_np_or_tensor, thres=0.001):
    for m in list_of_np_or_tensor:
        if hasattr(m, 'numpy'):
            assert not np.all(m.numpy() < thres)
        else:
            assert not np.all(m < thres)


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
        self.num_rollout_list = list(range(30))
        self.num_rollout_list_for_q_estimation = list(range(100))[1:]

        self.model = EnvironmentModel()
        self.preprocessor = Preprocessor(obs_space, self.args.obs_normalize, self.args.reward_preprocess_type,
                                         self.args.reward_scale_factor, gamma=self.args.gamma)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.w_timer = TimerStat()
        self.stats = {}
        self.reduced_num_minibatch = 4
        assert self.args.mini_batch_size % self.reduced_num_minibatch == 0

    def get_stats(self):
        return self.stats

    def get_batch_data(self, batch_data):
        self.batch_data = self.post_processing(batch_data)
        batch_advs, batch_tdlambda_returns = self.compute_advantage()
        batch_one_step_td_target = self.compute_one_step_td_target()

        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    batch_one_step_td_target=batch_one_step_td_target))
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

    def compute_one_step_td_target(self):
        processed_batch_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()
        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()

        act_tp1, _ = self.policy_with_value.compute_action(processed_batch_obs_tp1)
        batch_values_tp1 = self.policy_with_value.compute_Q_targets(processed_batch_obs_tp1,
                                                                    act_tp1.numpy())[0].numpy()[:, 0]

        batch_one_step_td_target = processed_batch_rewards + self.args.gamma * batch_values_tp1
        return batch_one_step_td_target

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])
        processed_batch_obs = self.preprocessor.tf_process_obses(self.batch_data['batch_obs']).numpy()
        processed_batch_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()

        batch_values = self.policy_with_value.compute_Q_targets(processed_batch_obs,
                                                                self.batch_data['batch_actions'])[0].numpy()[:, 0]
        act_tp1, _ = self.policy_with_value.compute_action(processed_batch_obs_tp1)
        batch_values_tp1 = self.policy_with_value.compute_Q_targets(processed_batch_obs_tp1,
                                                                    act_tp1.numpy())[0].numpy()[:, 0]

        batch_advs = np.zeros_like(self.batch_data['batch_rewards'], dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t + 1]
            if nextnonterminal < 0.1:
                delta = processed_batch_rewards[t] + self.args.gamma * batch_values_tp1[t] - batch_values[t]
            else:
                delta = processed_batch_rewards[t] + self.args.gamma * batch_values[t + 1] - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values
        return batch_advs, batch_tdlambda_returns

    def model_rollout_for_q_estimation(self, start_obses, start_actions):
        model = EnvironmentModel()
        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        actions_tile = self.tf.tile(start_actions, [self.M, 1])

        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0], 1))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0], 1))]

        model.reset(obses_tile)
        max_num_rollout = max(self.num_rollout_list_for_q_estimation)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards = model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * self.tf.reshape(processed_rewards, (-1, 1))
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0], 1)))

        all_Qs = self.policy_with_value.compute_Qs(
            self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[0]
        all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
        all_gammas = self.tf.concat(gammas_list, 0)

        final = self.tf.reshape(all_rewards_sums + all_gammas * all_Qs, (max_num_rollout + 1, self.M, -1))
        all_model_returns = self.tf.reduce_mean(final, axis=1)
        selected_model_returns = []
        for num_rollout in self.num_rollout_list_for_q_estimation:
            selected_model_returns.append(all_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        return self.tf.stop_gradient(selected_model_returns_flatten)

    def model_rollout_return(self, start_obses):
        processed_start_obses = self.preprocessor.tf_process_obses(start_obses)
        start_actions, _ = self.policy_with_value.compute_action(processed_start_obses)
        # judge_is_nan(start_actions)

        max_num_rollout = max(self.num_rollout_list)

        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        actions_tile = self.tf.tile(start_actions, [self.M, 1])
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0], 1))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0], 1))]

        self.model.reset(obses_tile)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * self.tf.reshape(processed_rewards, (-1, 1))
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_for_rollout.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0], 1)))

        all_Qs = self.policy_for_rollout.compute_Qs(
            self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[0]
        all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
        all_gammas = self.tf.concat(gammas_list, 0)

        final = self.tf.reshape(all_rewards_sums + all_gammas * all_Qs, (max_num_rollout + 1, self.M, -1))
        # final [[[time0+traj0], [time0+traj1], ..., [time0+trajn]],
        #        [[time1+traj0], [time1+traj1], ..., [time1+trajn]],
        #        ...
        #        [[timen+traj0], [timen+traj1], ..., [timen+trajn]],
        #        ]
        all_model_returns = self.tf.reduce_mean(final, axis=1)
        interval = int(self.args.mini_batch_size / self.reduced_num_minibatch)
        all_reduced_model_returns = self.tf.stack(
            [self.tf.reduce_mean(all_model_returns[:, i * interval:(i + 1) * interval], axis=-1) for i in
             range(self.reduced_num_minibatch)], axis=1)

        selected_model_returns, selected_reduced_model_returns = [], []
        for num_rollout in self.num_rollout_list:
            selected_model_returns.append(all_model_returns[num_rollout])
            selected_reduced_model_returns.append(all_reduced_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        selected_reduced_model_returns_flatten = self.tf.concat(selected_reduced_model_returns, 0)
        return selected_model_returns_flatten, selected_reduced_model_returns_flatten

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, data_target):
        targets = [data_target]
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        with self.tf.GradientTape() as tape:
            q_pred = self.policy_with_value.compute_Qs(processed_mb_obs, mb_actions)[0]
            with tape.stop_recording():
                bias_list = [self.tf.reduce_mean(self.tf.square(q_pred - data_target))]
                if len(self.num_rollout_list_for_q_estimation) > 0:
                    model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
                    for i in range(len(self.num_rollout_list_for_q_estimation)):
                        model_target_i = model_targets[i * self.args.mini_batch_size:
                                                       (i + 1) * self.args.mini_batch_size]
                        targets.append(model_target_i)
                        bias_list.append(self.tf.reduce_mean(self.tf.square(model_target_i - q_pred)
                                                             + self.tf.square(model_target_i- data_target)))
                epsilon = 1e-8
                bias_inverse_sum = self.tf.reduce_sum(
                    list(map(lambda x: 1. / (x + epsilon), bias_list)))
                w_bias_list = list(
                    map(lambda x: (1. / (x + epsilon)) / bias_inverse_sum, bias_list))
            q_loss = 0.5 * self.tf.reduce_sum(list(map(lambda w, target:
                                                       w * self.tf.reduce_mean(self.tf.square(q_pred - target)),
                                                       w_bias_list, targets)))
        q_gradient = tape.gradient(q_loss, self.policy_with_value.models[0].trainable_weights)
        return model_targets, w_bias_list, q_gradient

    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape(persistent=True) as tape:
            model_returns, reduced_model_returns = self.model_rollout_return(mb_obs)

        jaco = tape.jacobian(reduced_model_returns,
                             self.policy_with_value.policy.trainable_weights,
                             # unconnected_gradients=self.tf.UnconnectedGradients.ZERO,
                             experimental_use_pfor=True)
        # shape is len(self.policy_with_value.models[1].trainable_weights) * len(model_returns)
        # [[dy1/dx1, dy2/dx1,...(rolloutnum1)|dy1/dx1, dy2/dx1,...(rolloutnum2)| ...],
        #  [dy1/dx2, dy2/dx2, ...(rolloutnum1)|dy1/dx2, dy2/dx2,...(rolloutnum2)| ...],
        #  ...]
        return model_returns, reduced_model_returns, jaco

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        processed_mb_obs = self.preprocessor.np_process_obses(mb_obs)
        mb_advs = self.batch_data['batch_advs'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        mb_neglogps = self.batch_data['batch_neglogps'][start_idx: end_idx]
        # judge_is_nan([mb_obs])
        # judge_is_nan([processed_mb_obs])
        # judge_is_nan([mb_advs])
        # judge_is_nan([mb_tdlambda_returns])
        # judge_is_nan([mb_actions])
        # judge_is_nan([mb_neglogps])

        # print(self.preprocessor.get_params())

        with self.q_gradient_timer:
            # with self.tf.GradientTape() as tape:
            #     q_pred = self.policy_with_value.compute_Qs(processed_mb_obs, mb_actions)[0]
            #     q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - mb_tdlambda_returns))
            # q_gradient = tape.gradient(q_loss,
            #                            self.policy_with_value.models[0].trainable_weights,
            #                            # unconnected_gradients=self.tf.UnconnectedGradients.ZERO,
            #                            )  # TODO
            model_targets, w_q_list, q_gradient = self.q_forward_and_backward(mb_obs, mb_actions, mb_tdlambda_returns)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
            model_returns, reduced_model_returns, jaco = self.policy_forward_and_backward(mb_obs)
        # print(jaco, type(jaco))
        # print(model_returns, type(model_returns))
        # judge_is_nan([model_returns])
        # judge_is_nan(jaco)

        policy_gradient_list = []
        heuristic_bias_list = []
        var_list = []

        for rollout_index in range(len(self.num_rollout_list)):
            jaco_for_this_rollout = list(map(lambda x: x[rollout_index * self.reduced_num_minibatch:
                                                         (rollout_index + 1) * self.reduced_num_minibatch], jaco))

            gradient_std = []
            gradient_mean = []
            var = 0.
            for x in jaco_for_this_rollout:
                gradient_std.append(self.tf.math.reduce_std(x, 0))
                gradient_mean.append(self.tf.reduce_mean(x, 0))
                var += self.tf.reduce_mean(self.tf.square(gradient_std[-1])).numpy()
            heuristic_bias = self.tf.reduce_mean(
                self.tf.square(model_returns[rollout_index * self.args.mini_batch_size:
                                             (rollout_index + 1) * self.args.mini_batch_size]
                               - mb_tdlambda_returns)).numpy()

            # judge_is_nan(gradient_mean)

            policy_gradient_list.append(gradient_mean)
            heuristic_bias_list.append(heuristic_bias)
            var_list.append(var)
            # judge_is_nan(var_list)
            # judge_is_nan(heuristic_bias_list)

        w_heur_bias_list = []
        w_var_list = []
        epsilon = 1e-8
        heuristic_bias_inverse_sum = self.tf.reduce_sum(
            list(map(lambda x: 1. / (x + epsilon), heuristic_bias_list))).numpy()
        var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1. / (x + epsilon), var_list))).numpy()

        w_heur_bias_list = list(
            map(lambda x: (1. / (x + epsilon)) / heuristic_bias_inverse_sum, heuristic_bias_list))
        w_var_list = list(map(lambda x: (1. / (x + epsilon)) / var_inverse_sum, var_list))

        w_list = list(map(lambda x, y: (x + y) / 2., w_heur_bias_list, w_var_list))

        # judge_is_nan(w_list)

        final_policy_gradient = []
        for i in range(len(policy_gradient_list[0])):
            tmp = 0
            for j in range(len(policy_gradient_list)):
                # judge_is_nan(policy_gradient_list[j])
                tmp += w_list[j] * policy_gradient_list[j][i]
            final_policy_gradient.append(tmp)

        # judge_is_nan(q_gradient)
        # judge_is_nan(final_policy_gradient)

        final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(final_policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(num_traj_rollout=self.M,
                               num_rollout_list=self.num_rollout_list,
                               q_timer=self.q_gradient_timer.mean,
                               pg_time=self.policy_gradient_timer.mean,
                               w_q_list=list(map(lambda x: x.numpy(), w_q_list)),
                               var_list=var_list,
                               heuristic_bias_list=heuristic_bias_list,
                               w_var_list=w_var_list,
                               w_heur_bias_list=w_heur_bias_list,
                               w_list=w_list,
                               q_gradient_norm=q_gradient_norm.numpy(),
                               policy_gradient_norm=policy_gradient_norm.numpy()))

        #     logger.info('')
        #     logger.info('policy gradient use {}s'.format(self.policy_gradient_timer.mean))
        #     logger.info('var_list: {}'.format(list(map(lambda x: x.numpy(), var_list))))
        #     logger.info('heuristic_bias_list: {}'.format(list(map(lambda x: x.numpy(), heuristic_bias_list))))
        #     logger.info('w_var list: {}'.format(list(map(lambda x: x.numpy(), w_var_list))))
        #     logger.info('w_heur_bias list: {}'.format(list(map(lambda x: x.numpy(), w_heur_bias_list))))
        #     logger.info('final w list: {}'.format(list(map(lambda x: x.numpy(), w_list))))
        #     logger.info('q_gradient_norm: {}'.format(q_gradient_norm))
        #     logger.info('policy_gradient_norm: {}'.format(policy_gradient_norm))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
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
