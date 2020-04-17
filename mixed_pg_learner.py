import numpy as np
import gym
from gym.envs.user_defined.path_tracking_env import VehicleDynamics, EnvironmentModel
from preprocessor import Preprocessor
import time
import logging
from collections import deque
from utils.misc import safemean

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# logger.setLevel(logging.INFO)

# def judge_is_nan(list_of_np_or_tensor):
#     for m in list_of_np_or_tensor:
#         if hasattr(m, 'numpy'):
#             if np.any(np.isnan(m.numpy())):
#                 print(list_of_np_or_tensor)
#                 raise ValueError
#         else:
#             if np.any(np.isnan(m)):
#                 print(list_of_np_or_tensor)
#                 raise ValueError
#
#
# def judge_less_than(list_of_np_or_tensor, thres=0.001):
#     for m in list_of_np_or_tensor:
#         if hasattr(m, 'numpy'):
#             assert not np.all(m.numpy() < thres)
#         else:
#             assert not np.all(m < thres)


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
        self.sample_num_in_learner = self.args.sample_num_in_learner
        self.batch_size = self.args.num_agent * self.args.sample_n_step
        self.env = gym.make(self.args.env_id, num_agent=self.batch_size)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.batch_data = {}
        self.all_data = {}
        self.policy_for_rollout = policy_cls(obs_space, act_space, self.args)
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update
        self.num_rollout_list_for_q_estimation = self.args.num_rollout_list_for_q_estimation

        self.model = EnvironmentModel()
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma, num_agent=self.args.num_agent)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.stats = {}
        self.reduced_num_minibatch = 4
        assert self.args.mini_batch_size % self.reduced_num_minibatch == 0

    def get_stats(self):
        return self.stats

    def get_batch_data111(self, batch_data, epinfos):
        self.batch_data = self.post_processing(batch_data)
        batch_advs, batch_tdlambda_returns = self.compute_advantage()

        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    ))
        self.flatten_and_shuffle()

    def flatten_and_shuffle(self):
        permutation = np.random.permutation(self.batch_size)
        for key, val in self.batch_data.items():
            val_reshape = val.reshape(self.batch_size, -1)
            if val_reshape.shape[1] == 1:
                val_reshape = val_reshape[:, 0]
            self.batch_data[key] = val_reshape[permutation]

    def get_batch_data(self, start_data, epinfos):
        self.all_data = self.post_processing2(start_data)
        batch_obs, batch_actions = self.all_data['all_obs'][0], self.all_data['all_actions'][0]
        # batch_advs, batch_tdlambda_returns = self.compute_advantage2()

        all_n_step_target, n_step_bias = self.compute_n_step_target_and_bias()

        self.batch_data.update(dict(batch_obs=batch_obs,
                                    batch_actions=batch_actions,
                                    all_n_step_target=all_n_step_target,
                                    n_step_bias=n_step_bias,
                                    ))
        self.shuffle()

        # print(self.batch_data['batch_obs'].shape)  # batch_size * obs_dim
        # print(self.batch_data['batch_actions'].shape)  # batch_size * act_dim
        # print(self.batch_data['batch_advs'].shape)  # batch_size,
        # print(self.batch_data['batch_tdlambda_returns'].shape)  # batch_size,

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               'batch_neglogps': np.asarray(list(map(lambda x: x[5], batch_data)), dtype=np.float32)}
        return tmp

    def post_processing2(self, start_data):
        start_obs = np.asarray(list(map(lambda x: x[0], start_data)), dtype=np.float32).reshape((self.batch_size, -1))
        all_data = self.sample(start_obs)
        tmp = {'all_obs': np.asarray(list(map(lambda x: x[0], all_data)), dtype=np.float32),
               'all_actions': np.asarray(list(map(lambda x: x[1], all_data)), dtype=np.float32),
               'all_rewards': np.asarray(list(map(lambda x: x[2], all_data)), dtype=np.float32),
               'all_obs_tp1': np.asarray(list(map(lambda x: x[3], all_data)), dtype=np.float32),
               'all_dones': np.asarray(list(map(lambda x: x[4], all_data)), dtype=np.float32),
               'all_neglogps': np.asarray(list(map(lambda x: x[5], all_data)), dtype=np.float32)
               }

        # print('all_rewards', tmp['all_rewards'].shape)  # sample_num_in_learner * batch_size
        # print('all_obs', tmp['all_obs'].shape)  # sample_num_in_learner * batch_size * dim_obs
        # print('all_actions', tmp['all_actions'].shape)  # sample_num_in_learner * batch_size * dim_act
        # print('all_obs_tp1', tmp['all_obs_tp1'].shape)  # sample_num_in_learner * batch_size * dim_obs
        # print('all_dones', tmp['all_dones'].shape)  # sample_num_in_learner * batch_size
        # print('all_neglogps', tmp['all_neglogps'].shape)  # sample_num_in_learner * batch_size

        return tmp

    def sample(self, start_obs):
        batch_data = []
        obs = start_obs
        self.env.reset(init_obs=obs)
        for _ in range(self.sample_num_in_learner):
            processed_obs = self.preprocessor.tf_process_obses(obs).numpy()
            action, neglogp = self.policy_with_value.compute_action(processed_obs)
            obs_tp1, reward, _, info = self.env.step(action.numpy())
            done = np.zeros((self.batch_size,), dtype=np.int)
            batch_data.append((obs, action.numpy(), reward, obs_tp1, done, neglogp.numpy()))
            obs = obs_tp1.copy()

        return batch_data

    def compute_n_step_target_and_bias(self):
        # print(self.all_data['all_rewards'].shape)  # sample_num_in_learner * batch_size
        # print(self.all_data['all_obs'].shape)  # sample_num_in_learner * batch_size * dim_obs
        # print(self.all_data['all_actions'].shape)  # sample_num_in_learner * batch_size * dim_act
        # print(self.all_data['all_obs_tp1'].shape)  # sample_num_in_learner * batch_size * dim_obs
        # print(self.all_data['all_dones'].shape)  # sample_num_in_learner * batch_size
        # print(self.all_data['all_neglogps'].shape)  # sample_num_in_learner * batch_size

        processed_all_obs = self.preprocessor.tf_process_obses(
            self.all_data['all_obs']).numpy()  # sample_num_in_learner * batch_size * obs_dim
        processed_all_obs_tp1 = self.preprocessor.tf_process_obses(self.all_data['all_obs_tp1']).numpy()

        processed_all_rewards = self.preprocessor.tf_process_rewards(self.all_data['all_rewards']).numpy()

        all_values = \
            self.policy_with_value.compute_Q_target(
                processed_all_obs.reshape(self.sample_num_in_learner * self.batch_size, -1),
                self.all_data['all_actions'].reshape(self.sample_num_in_learner * self.batch_size,
                                                     -1)).numpy().reshape(self.sample_num_in_learner, self.batch_size)
        act_tp1, _ = self.policy_with_value.compute_action(
            processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1))
        all_values_tp1 = \
            self.policy_with_value.compute_Q_target(processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1),
                                                    act_tp1.numpy()).numpy().reshape(self.sample_num_in_learner, self.batch_size)
        all_n_step_target = np.zeros((self.sample_num_in_learner+1, self.batch_size), dtype=np.float32)
        all_n_step_target[0] = all_values[0]
        for t in range(1, self.sample_num_in_learner+1):
            last_values = all_values_tp1[t-1] if t == self.sample_num_in_learner else all_values[t]
            all_n_step_target[t] = all_n_step_target[t-1] - pow(self.args.gamma, t-1) * all_values[t-1] +\
                                   pow(self.args.gamma, t-1) * processed_all_rewards[t-1] + \
                                   pow(self.args.gamma, t) * last_values
        n_step_bias = np.abs(all_n_step_target - all_n_step_target[-1])
        all_n_step_target = np.transpose(all_n_step_target)  # self.batch_size * self.sample_num_in_learner+1
        n_step_bias = np.transpose(n_step_bias)  # self.batch_size * self.sample_num_in_learner+1

        return all_n_step_target, n_step_bias

    def shuffle(self):
        permutation = np.random.permutation(self.batch_size)
        for key, val in self.batch_data.items():
            self.batch_data[key] = val[permutation]

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])  # n_step * num_agent
        # print(self.batch_data['batch_rewards'].shape)
        # print(self.batch_data['batch_obs'].shape)
        # print(self.batch_data['batch_actions'].shape)
        # print(self.batch_data['batch_obs_tp1'].shape)
        # print(self.batch_data['batch_dones'].shape)
        # print(self.batch_data['batch_neglogps'].shape)

        processed_batch_obs = self.preprocessor.tf_process_obses(
            self.batch_data['batch_obs']).numpy()  # # n_step * num_agent * obs_dim
        processed_batch_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()

        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()

        batch_values = \
            self.policy_with_value.compute_Q_target(processed_batch_obs.reshape(n_steps * self.args.num_agent, -1),
                                                    self.batch_data['batch_actions'].reshape(
                                                        n_steps * self.args.num_agent,
                                                        -1)).numpy().reshape(
                n_steps, self.args.num_agent)
        act_tp1, _ = self.policy_with_value.compute_action(
            processed_batch_obs_tp1.reshape(n_steps * self.args.num_agent, -1))
        batch_values_tp1 = \
            self.policy_with_value.compute_Q_target(processed_batch_obs_tp1.reshape(n_steps * self.args.num_agent, -1),
                                                    act_tp1.numpy()).numpy().reshape(n_steps, self.args.num_agent)

        batch_advs = np.zeros_like(self.batch_data['batch_rewards'], dtype=np.float32)
        lastgaelam = np.zeros_like(self.batch_data['batch_rewards'][0, :], dtype=np.float32)
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t]
            delta = processed_batch_rewards[t] + self.args.gamma * np.where(nextnonterminal < 0.1, batch_values_tp1[t],
                                                                            batch_values[t + 1]) - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values
        return batch_advs, batch_tdlambda_returns

    def compute_advantage2(self):  # require data is in order
        # print(self.all_data['all_rewards'].shape)  # sample_num_in_learner * batch_size
        # print(self.all_data['all_obs'].shape)  # sample_num_in_learner * batch_size * dim_obs
        # print(self.all_data['all_actions'].shape)  # sample_num_in_learner * batch_size * dim_act
        # print(self.all_data['all_obs_tp1'].shape)  # sample_num_in_learner * batch_size * dim_obs
        # print(self.all_data['all_dones'].shape)  # sample_num_in_learner * batch_size
        # print(self.all_data['all_neglogps'].shape)  # sample_num_in_learner * batch_size

        processed_all_obs = self.preprocessor.tf_process_obses(
            self.all_data['all_obs']).numpy()  # sample_num_in_learner * batch_size * obs_dim
        # processed_all_obs_tp1 = self.preprocessor.tf_process_obses(self.all_data['all_obs_tp1']).numpy()

        processed_all_rewards = self.preprocessor.tf_process_rewards(self.all_data['all_rewards']).numpy()

        all_values = \
            self.policy_with_value.compute_Q_target(processed_all_obs.reshape(self.sample_num_in_learner * self.batch_size, -1),
                                                    self.all_data['all_actions'].reshape(
                                                        self.sample_num_in_learner * self.batch_size,
                                                        -1)).numpy().reshape(
                self.sample_num_in_learner, self.batch_size)
        # act_tp1, _ = self.policy_with_value.compute_action(
        #     processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1))
        # all_values_tp1 = \
        #     self.policy_with_value.compute_Q_target(processed_all_obs_tp1.reshape(self.sample_num_in_learner * self.batch_size, -1),
        #                                             act_tp1.numpy()).numpy().reshape(self.sample_num_in_learner, self.batch_size)

        all_advs = np.zeros_like(self.all_data['all_rewards'], dtype=np.float32)
        lastgaelam = np.zeros_like(self.all_data['all_rewards'][0, :], dtype=np.float32)
        for t in reversed(range(self.sample_num_in_learner-1)):
            delta = processed_all_rewards[t] + self.args.gamma * all_values[t+1] - all_values[t]
            all_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * lastgaelam
        all_tdlambda_returns = all_advs + all_values
        return all_advs[0], all_tdlambda_returns[0]

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    # @tf.function
    # def model_based_q_forward_and_backward(self, mb_obs, mb_action):
    #     with self.tf.GradientTape() as tape:
    #         obses = mb_obs
    #         self.model.reset(obses)
    #         processed_obses = self.preprocessor.tf_process_obses(obses)
    #         actions = mb_action
    #         q_pred = self.policy_with_value.compute_Q(processed_obses, actions)[:, 0]
    #         reward_sum = self.tf.zeros((obses.shape[0],))
    #         for i in range(30):
    #             obses, rewards = self.model.rollout_out(actions)
    #             processed_rewards = self.preprocessor.tf_process_rewards(rewards)
    #             reward_sum += self.tf.pow(self.args.gamma, i) * processed_rewards
    #             processed_obses = self.preprocessor.tf_process_obses(obses)
    #             actions, _ = self.policy_with_value.compute_action(processed_obses)
    #
    #         Qs = self.policy_with_value.compute_Q(processed_obses, actions)[:, 0]
    #         target = self.tf.stop_gradient(reward_sum + self.tf.pow(self.args.gamma, 30) * Qs)
    #         q_loss = self.tf.reduce_mean(self.tf.square(target - q_pred))
    #
    #     q_gradient = tape.gradient(q_loss, self.policy_with_value.Q.trainable_weights)
    #     return q_gradient, q_loss

    # @tf.function
    # def model_based_policy_forward_and_backward(self, mb_obs):
    #     with self.tf.GradientTape() as tape:
    #         obses = mb_obs
    #         self.model.reset(obses)
    #         reward_sum = self.tf.zeros((obses.shape[0],))
    #         for i in range(30):
    #             processed_obses = self.preprocessor.tf_process_obses(obses)
    #             actions, _ = self.policy_with_value.compute_action(processed_obses)
    #             obses, rewards = self.model.rollout_out(actions)
    #             processed_rewards = self.preprocessor.tf_process_rewards(rewards)
    #             reward_sum += self.tf.pow(self.args.gamma, i) * processed_rewards
    #
    #         processed_obses = self.preprocessor.tf_process_obses(obses)
    #         actions, _ = self.policy_with_value.compute_action(processed_obses)
    #         Qs = self.policy_with_value.compute_Q(processed_obses, actions)[:, 0]
    #         target = reward_sum + self.tf.pow(self.args.gamma, 30) * Qs
    #         policy_loss = -self.tf.reduce_mean(target)
    #
    #     policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)
    #     return policy_gradient, policy_loss

    # @tf.function
    # def model_based_policy_forward_and_backward(self, mb_obs):
    #     self.obs_forward = []
    #     self.action_forward = []
    #     self.reward_forward = []
    #     self.forward_step = 30
    #     for i in range(self.forward_step):
    #         self.obs_forward.append([])
    #         self.action_forward.append([])
    #         self.reward_forward.append([])
    #     self.obs_forward.append([])
    #     with self.tf.GradientTape() as tape:
    #         for i in range(self.forward_step):
    #             if i == 0:
    #                 self.obs_forward[i] = mb_obs
    #                 action, _ = self.policy_with_value.compute_action(self.preprocessor.tf_process_obses(self.obs_forward[i]))
    #                 self.action_forward[i] = np.stack([action[:, 0] * np.pi / 9, action[:, 1] * 2], 1)
    #
    #                 self.obs_forward[i + 1] = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
    #                 self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])
    #             else:
    #                 action, _ = self.policy_with_value.compute_action(self.preprocessor.tf_process_obses(self.obs_forward[i]))
    #                 self.action_forward[i] = np.stack([action[:, 0] * np.pi / 9, action[:, 1] * 2], 1)
    #                 self.obs_forward[i + 1] = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
    #                 self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])
    #
    #         action_next, _ = self.policy_with_value.compute_action(self.preprocessor.tf_process_obses(self.obs_forward[-1]))
    #         q_next = self.policy_with_value.compute_Q(self.preprocessor.tf_process_obses(self.obs_forward[-1]),
    #                                                   action_next)[:, 0]
    #         target = self.tf.zeros_like(q_next)
    #         for i in range(self.forward_step):
    #             target += self.tf.pow(self.args.gamma, i) * self.reward_forward[i]
    #         target += self.tf.pow(self.args.gamma, self.forward_step) * q_next
    #         policy_loss = -self.tf.reduce_mean(target)
    #
    #     policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)
    #     return policy_gradient, policy_loss

    # @tf.function
    # def model_based_q_forward_and_backward(self, mb_obs, mb_action):
    #     self.obs_forward = []
    #     self.action_forward = []
    #     self.reward_forward = []
    #     self.forward_step = 30
    #     for i in range(self.forward_step):
    #         self.obs_forward.append([])
    #         self.action_forward.append([])
    #         self.reward_forward.append([])
    #     self.obs_forward.append([])
    #     with self.tf.GradientTape() as tape:
    #         for i in range(self.forward_step):
    #             if i == 0:
    #                 self.obs_forward[i] = mb_obs
    #                 self.action_forward[i] = np.stack([mb_action[:, 0] * np.pi / 9, mb_action[:, 1] * 2], 1)
    #                 self.obs_forward[i+1] = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
    #                 self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])
    #             else:
    #                 action, _ = self.policy_with_value.compute_action(self.preprocessor.tf_process_obses(self.obs_forward[i]))
    #                 self.action_forward[i] = np.stack([action[:, 0] * np.pi / 9, action[:, 1] * 2], 1)
    #                 self.obs_forward[i + 1] = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
    #                 self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])
    #
    #         action_next, _ = self.policy_with_value.compute_action(self.preprocessor.tf_process_obses(self.obs_forward[-1]))
    #         q_next = self.policy_with_value.compute_Q(self.preprocessor.tf_process_obses(self.obs_forward[-1]),
    #                                                   action_next)[:, 0]
    #         target = self.tf.zeros_like(q_next)
    #         for i in range(self.forward_step):
    #             target += self.tf.pow(self.args.gamma, i) * self.reward_forward[i]
    #         target += self.tf.pow(self.args.gamma, self.forward_step) * q_next
    #         q_pred = self.policy_with_value.compute_Q(self.preprocessor.process_obs(mb_obs),
    #                                                   mb_action)[:, 0]
    #         q_loss = self.tf.reduce_mean(self.tf.square(self.tf.stop_gradient(target)-q_pred))
    #
    #     q_gradient = tape.gradient(q_loss, self.policy_with_value.Q.trainable_weights)
    #     return q_gradient, q_loss

    def model_rollout_for_q_estimation(self, start_obses, start_actions):
        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        actions_tile = self.tf.tile(start_actions, [self.M, 1])

        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0],))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0],))]

        self.model.reset(obses_tile)
        max_num_rollout = max(self.num_rollout_list_for_q_estimation)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * processed_rewards
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_Qs = self.policy_with_value.compute_Q_target(
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
            all_rewards_sums = self.tf.concat(rewards_sum_list, 0)
            all_gammas = self.tf.concat(gammas_list, 0)
            all_targets = all_rewards_sums + all_gammas * all_Qs

            final = self.tf.reshape(all_targets, (max_num_rollout + 1, self.M, -1))
            all_model_returns = self.tf.reduce_mean(final, axis=1)
        selected_model_returns = []
        for num_rollout in self.num_rollout_list_for_q_estimation:
            selected_model_returns.append(all_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        return self.tf.stop_gradient(selected_model_returns_flatten)

    def model_rollout_for_policy_update(self, start_obses):
        processed_start_obses = self.preprocessor.tf_process_obses(start_obses)
        start_actions, _ = self.policy_with_value.compute_action(processed_start_obses)
        # judge_is_nan(start_actions)

        max_num_rollout = max(self.num_rollout_list_for_policy_update)

        obses_tile = self.tf.tile(start_obses, [self.M, 1])
        processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
        actions_tile = self.tf.tile(start_actions, [self.M, 1])
        processed_obses_tile_list = [processed_obses_tile]
        actions_tile_list = [actions_tile]
        rewards_sum_tile = self.tf.zeros((obses_tile.shape[0],))
        rewards_sum_list = [rewards_sum_tile]
        gammas_list = [self.tf.ones((obses_tile.shape[0],))]

        self.model.reset(obses_tile)
        if max_num_rollout > 0:
            for ri in range(max_num_rollout):
                obses_tile, rewards = self.model.rollout_out(actions_tile)
                processed_obses_tile = self.preprocessor.tf_process_obses(obses_tile)
                processed_rewards = self.preprocessor.tf_process_rewards(rewards)
                rewards_sum_tile += self.tf.pow(self.args.gamma, ri) * processed_rewards
                rewards_sum_list.append(rewards_sum_tile)
                actions_tile, _ = self.policy_for_rollout.compute_action(processed_obses_tile) if not \
                    self.args.deriv_interval_policy else self.policy_with_value.compute_action(processed_obses_tile)
                processed_obses_tile_list.append(processed_obses_tile)
                actions_tile_list.append(actions_tile)
                gammas_list.append(self.tf.pow(self.args.gamma, ri + 1) * self.tf.ones((obses_tile.shape[0],)))

        with self.tf.name_scope('compute_all_model_returns') as scope:
            all_Qs = self.policy_with_value.compute_Q(
                self.tf.concat(processed_obses_tile_list, 0), self.tf.concat(actions_tile_list, 0))[:, 0]
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

        selected_model_returns, minus_selected_reduced_model_returns = [], []
        for num_rollout in self.num_rollout_list_for_policy_update:
            selected_model_returns.append(all_model_returns[num_rollout])
            minus_selected_reduced_model_returns.append(-all_reduced_model_returns[num_rollout])

        selected_model_returns_flatten = self.tf.concat(selected_model_returns, 0)
        minus_selected_reduced_model_returns_flatten = self.tf.concat(minus_selected_reduced_model_returns, 0)
        value_mean = self.tf.reduce_mean(all_model_returns[0])
        return selected_model_returns_flatten, minus_selected_reduced_model_returns_flatten, value_mean

    # @tf.function
    # def q_forward_and_backward(self, mb_obs, mb_actions, data_target):
    #     processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
    #     with self.tf.GradientTape() as tape:
    #         with self.tf.name_scope('q_loss') as scope:
    #             q_pred = self.policy_with_value.compute_Q(processed_mb_obs, mb_actions)[:, 0]
    #             with tape.stop_recording():
    #                 targets = []
    #                 bias_list = []
    #                 model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
    #                 for i, num_rollout in enumerate(self.num_rollout_list_for_q_estimation):
    #                     model_target_i = model_targets[i * self.args.mini_batch_size:
    #                                                    (i + 1) * self.args.mini_batch_size]
    #                     if num_rollout == 0:
    #                         targets.append(data_target)
    #                         bias_list.append(self.tf.reduce_mean(self.tf.square(q_pred - data_target)))
    #                     else:
    #                         targets.append(model_target_i)
    #                         bias_list.append(self.tf.reduce_mean(self.tf.square(model_target_i - data_target)))
    #                         # bias_list.append(self.tf.reduce_mean(self.tf.square(model_target_i - q_pred)
    #                         #                                      + self.tf.square(model_target_i - data_target)))
    #                 epsilon = 1e-8
    #                 bias_inverse_sum = self.tf.reduce_sum(
    #                     list(map(lambda x: 1. / (x + epsilon), bias_list)))
    #                 w_bias_list = list(
    #                     map(lambda x: (1. / (x + epsilon)) / bias_inverse_sum, bias_list))
    #             final_target = self.tf.reduce_sum(list(map(lambda w, target: w * target, w_bias_list, targets)), axis=0)
    #             q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - final_target))
    #     with self.tf.name_scope('q_gradient') as scope:
    #         q_gradient = tape.gradient(q_loss, self.policy_with_value.Q.trainable_weights)
    #     return model_targets, w_bias_list, q_gradient, q_loss, q_pred

    @tf.function
    def q_forward_and_backward(self, mb_obs, mb_actions, mb_n_step_targets):
        processed_mb_obs = self.preprocessor.tf_process_obses(mb_obs)
        with self.tf.GradientTape() as tape:
            with self.tf.name_scope('q_loss') as scope:
                q_pred = self.policy_with_value.compute_Q(processed_mb_obs, mb_actions)[:, 0]
                q_loss = 0.5 * self.tf.reduce_mean(self.tf.square(q_pred - mb_n_step_targets[:, -1]))

        with self.tf.name_scope('q_gradient') as scope:
            q_gradient = tape.gradient(q_loss, self.policy_with_value.Q.trainable_weights)
        model_targets = self.model_rollout_for_q_estimation(mb_obs, mb_actions)
        model_bias_list = []
        for i, num_rollout in enumerate(self.num_rollout_list_for_q_estimation):
            model_target_i = model_targets[i * self.args.mini_batch_size:
                                           (i + 1) * self.args.mini_batch_size]
            data_target_i = mb_n_step_targets[:, num_rollout]
            if i == 0:
                self.tf.print(self.tf.reduce_mean(self.tf.abs(model_target_i-data_target_i)))
            model_bias_list.append(self.tf.reduce_mean(self.tf.abs(model_target_i-data_target_i)))
        return model_targets, q_gradient, q_loss, model_bias_list


    @tf.function
    def policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape(persistent=True) as tape:
            model_returns, minus_reduced_model_returns, value_mean = self.model_rollout_for_policy_update(mb_obs)

        with self.tf.name_scope('policy_jacobian') as scope:
            jaco = tape.jacobian(minus_reduced_model_returns,
                                 self.policy_with_value.policy.trainable_weights,
                                 experimental_use_pfor=True)
            # shape is len(self.policy_with_value.models[1].trainable_weights) * len(model_returns)
            # [[dy1/dx1, dy2/dx1,...(rolloutnum1)|dy1/dx1, dy2/dx1,...(rolloutnum2)| ...],
            #  [dy1/dx2, dy2/dx2, ...(rolloutnum1)|dy1/dx2, dy2/dx2,...(rolloutnum2)| ...],
            #  ...]
            return model_returns, minus_reduced_model_returns, jaco, value_mean

    def export_graph111(self, writer):
        start_idx, end_idx = 0, self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_tdlambda_returns)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def export_graph(self, writer):
        start_idx, end_idx = 0, self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_all_n_step_target = self.batch_data['all_n_step_target'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.q_forward_and_backward(mb_obs, mb_actions, mb_all_n_step_target)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient_over_ith_minibatch111(self, i):  # compute gradient of the i-th mini-batch
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]



        with self.q_gradient_timer:
            model_targets, w_q_list, q_gradient, q_loss, q_pred = self.q_forward_and_backward(mb_obs, mb_actions,
                                                                                              mb_tdlambda_returns)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
            model_returns, minus_reduced_model_returns, jaco, value_mean = self.policy_forward_and_backward(mb_obs)

        policy_gradient_list = []
        heuristic_bias_list = []
        var_list = []
        final_policy_gradient = []

        for rollout_index in range(len(self.num_rollout_list_for_policy_update)):
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
                self.tf.square(model_targets[rollout_index * self.args.mini_batch_size:
                                             (rollout_index + 1) * self.args.mini_batch_size]
                               - mb_tdlambda_returns)).numpy()

            policy_gradient_list.append(gradient_mean)
            heuristic_bias_list.append(heuristic_bias)
            var_list.append(var)

        epsilon = 1e-8
        heuristic_bias_inverse_sum = self.tf.reduce_sum(
            list(map(lambda x: 1. / (x + epsilon), heuristic_bias_list))).numpy()
        var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1. / (x + epsilon), var_list))).numpy()

        w_heur_bias_list = list(
            map(lambda x: (1. / (x + epsilon)) / heuristic_bias_inverse_sum, heuristic_bias_list))
        w_var_list = list(map(lambda x: (1. / (x + epsilon)) / var_inverse_sum, var_list))

        w_list = list(map(lambda x, y: (x + y) / 2., w_heur_bias_list, w_var_list))

        for i in range(len(policy_gradient_list[0])):
            tmp = 0
            for j in range(len(policy_gradient_list)):
                # judge_is_nan(policy_gradient_list[j])
                tmp += w_list[j] * policy_gradient_list[j][i]
            final_policy_gradient.append(tmp)

        final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(final_policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(

            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            q_loss=q_loss.numpy(),
            value_mean=value_mean.numpy(),
            q_gradient_norm=q_gradient_norm.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
            num_traj_rollout=self.M,
            num_rollout_list=self.num_rollout_list_for_policy_update,
            w_q_list=list(map(lambda x: x.numpy(), w_q_list)),
            var_list=var_list,
            heuristic_bias_list=heuristic_bias_list,
            w_var_list=w_var_list,
            w_heur_bias_list=w_heur_bias_list,
            w_list=w_list
        ))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        mb_all_n_step_target = self.batch_data['all_n_step_target'][start_idx: end_idx]
        mb_n_step_all_bias = self.batch_data['n_step_bias'][start_idx: end_idx]
        data_bias_list = []
        for num_rollout in self.num_rollout_list_for_q_estimation:
            data_bias_list.append(np.mean(mb_n_step_all_bias[:, num_rollout]))
        base = data_bias_list[-1]
        data_bias_list = [b-base for b in data_bias_list]
        with self.q_gradient_timer:
            model_targets, q_gradient, q_loss, model_bias_list = self.q_forward_and_backward(mb_obs, mb_actions,
                                                                                             mb_all_n_step_target)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
            model_returns, minus_reduced_model_returns, jaco, value_mean = self.policy_forward_and_backward(mb_obs)

        model_bias_list = [a.numpy() for a in model_bias_list]
        print(model_bias_list)
        policy_gradient_list = []
        heuristic_bias_list = [a+b for a, b in zip(model_bias_list, data_bias_list)]
        var_list = []
        final_policy_gradient = []

        for rollout_index in range(len(self.num_rollout_list_for_policy_update)):
            jaco_for_this_rollout = list(map(lambda x: x[rollout_index * self.reduced_num_minibatch:
                                                         (rollout_index + 1) * self.reduced_num_minibatch], jaco))

            gradient_std = []
            gradient_mean = []
            var = 0.
            for x in jaco_for_this_rollout:
                gradient_std.append(self.tf.math.reduce_std(x, 0))
                gradient_mean.append(self.tf.reduce_mean(x, 0))
                var += self.tf.reduce_mean(self.tf.square(gradient_std[-1])).numpy()

            policy_gradient_list.append(gradient_mean)
            var_list.append(var)

        epsilon = 1e-8
        heuristic_bias_inverse_sum = self.tf.reduce_sum(
            list(map(lambda x: 1. / (x + epsilon), heuristic_bias_list))).numpy()
        var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1. / (x + epsilon), var_list))).numpy()

        w_heur_bias_list = list(
            map(lambda x: (1. / (x + epsilon)) / heuristic_bias_inverse_sum, heuristic_bias_list))
        w_var_list = list(map(lambda x: (1. / (x + epsilon)) / var_inverse_sum, var_list))

        # w_list = list(map(lambda x, y: (x + y) / 2., w_heur_bias_list, w_var_list))
        w_list = w_heur_bias_list

        for i in range(len(policy_gradient_list[0])):
            tmp = 0
            for j in range(len(policy_gradient_list)):
                # judge_is_nan(policy_gradient_list[j])
                tmp += w_list[j] * policy_gradient_list[j][i]
            final_policy_gradient.append(tmp)

        final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(final_policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(

            q_timer=self.q_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            q_loss=q_loss.numpy(),
            value_mean=value_mean.numpy(),
            q_gradient_norm=q_gradient_norm.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
            num_traj_rollout=self.M,
            num_rollout_list=self.num_rollout_list_for_policy_update,
            w_q_list=[],
            var_list=var_list,
            heuristic_bias_list=heuristic_bias_list,
            model_bias_list=model_bias_list,
            data_bias_list=data_bias_list,
            w_var_list=w_var_list,
            w_heur_bias_list=w_heur_bias_list,
            w_list=w_list
        ))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
        return list(map(lambda x: x.numpy(), gradient_tensor))


if __name__ == '__main__':
    pass
