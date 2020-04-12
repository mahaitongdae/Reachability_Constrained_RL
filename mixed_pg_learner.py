import numpy as np
import gym
from gym.envs.user_defined.path_tracking_env import VehicleDynamics
from preprocessor import Preprocessor
import time
import logging
from collections import deque
from utils.misc import safemean

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
        env = gym.make(self.args.env_id, num_agent=self.args.num_agent)
        obs_space, act_space = env.observation_space, env.action_space
        env.close()
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.policy_for_rollout = policy_cls(obs_space, act_space, self.args)
        self.batch_data = {}
        self.epinfos = {}
        self.M = 1
        self.num_rollout_list_for_policy_update = list(range(0, 31, 2)) if not self.args.model_based else [20]
        self.num_rollout_list_for_q_estimation = list(range(0, 31, 2))[1:] if not self.args.model_based else [20]

        self.model = VehicleDynamics()
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale_factor, self.args.reward_scale_factor,
                                         gamma=self.args.gamma, num_agent=self.args.num_agent)
        self.policy_gradient_timer = TimerStat()
        self.q_gradient_timer = TimerStat()
        self.w_timer = TimerStat()
        self.stats = {}
        self.reduced_num_minibatch = 4
        assert self.args.mini_batch_size % self.reduced_num_minibatch == 0
        self.epinfobuf = deque(maxlen=100)
        self.obs_forward = []
        self.action_forward = []
        self.reward_forward = []
        self.forward_step = 30
        for i in range(self.forward_step):
            self.obs_forward.append([])
            self.action_forward.append([])
            self.reward_forward.append([])
        self.obs_forward.append([])

    def get_stats(self):
        return self.stats

    def get_batch_data(self, batch_data, epinfos):
        self.batch_data = self.post_processing(batch_data)
        self.epinfobuf.extend(epinfos)
        eprewmean = safemean([epinfo['r'] for epinfo in self.epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in self.epinfobuf])
        self.stats.update(dict(eprewmean=eprewmean,
                               eplenmean=eplenmean))
        batch_advs, batch_tdlambda_returns = self.compute_advantage()
        batch_one_step_td_target = self.compute_one_step_td_target()

        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    batch_one_step_td_target=batch_one_step_td_target,
                                    ))
        self.flatten_and_shuffle()

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               'batch_neglogps': np.asarray(list(map(lambda x: x[5], batch_data)), dtype=np.float32)}
        return tmp

    def batch_data_count(self):
        return self.args.sample_n_step * self.args.num_agent

    def flatten_and_shuffle(self):
        permutation = np.random.permutation(self.batch_data_count())
        for key, val in self.batch_data.items():
            val_reshape = val.reshape(self.batch_data_count(), -1)
            self.batch_data[key] = val_reshape[permutation]

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def compute_one_step_td_target(self):
        processed_batch_obs_tp1 = self.preprocessor.tf_process_obses(self.batch_data['batch_obs_tp1']).numpy()
        processed_batch_rewards = self.preprocessor.tf_process_rewards(self.batch_data['batch_rewards']).numpy()

        act_tp1, _ = self.policy_with_value.compute_action(processed_batch_obs_tp1.reshape(self.args.sample_n_step * self.args.num_agent, -1))
        batch_values_tp1 = self.policy_with_value.compute_Q_targets(processed_batch_obs_tp1.reshape(self.args.sample_n_step * self.args.num_agent, -1),
                                                                    act_tp1.numpy())[0].numpy().reshape(self.args.sample_n_step, self.args.num_agent)

        batch_one_step_td_target = processed_batch_rewards + self.args.gamma * batch_values_tp1
        return batch_one_step_td_target

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
        self.policy_with_value.compute_Q_targets(processed_batch_obs.reshape(n_steps * self.args.num_agent, -1),
                                                 self.batch_data['batch_actions'].reshape(n_steps * self.args.num_agent,
                                                                                          -1))[0].numpy().reshape(
            n_steps, self.args.num_agent)
        act_tp1, _ = self.policy_with_value.compute_action(
            processed_batch_obs_tp1.reshape(n_steps * self.args.num_agent, -1))
        batch_values_tp1 = \
        self.policy_with_value.compute_Q_targets(processed_batch_obs_tp1.reshape(n_steps * self.args.num_agent, -1),
                                                 act_tp1.numpy())[0].numpy().reshape(n_steps, self.args.num_agent)

        batch_advs = np.zeros_like(self.batch_data['batch_rewards'], dtype=np.float32)
        lastgaelam = np.zeros_like(self.batch_data['batch_rewards'][0, :], dtype=np.float32)
        for t in reversed(range(n_steps-1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t]
            delta = processed_batch_rewards[t] + self.args.gamma * np.where(nextnonterminal < 0.1, batch_values_tp1[t],
                                                                            batch_values[t + 1]) - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values
        return batch_advs, batch_tdlambda_returns

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)


    # def model_based_q_forward_and_backward(self, mb_obs, mb_action):
    #     with self.tf.GradientTape() as tape:
    #         obses = mb_obs
    #         self.model.reset(obses)
    #         processed_obses = self.preprocessor.tf_process_obses(obses)
    #         actions = mb_action
    #         q_pred = self.policy_with_value.compute_Qs(processed_obses, actions)[0][:, 0]
    #         reward_sum = self.tf.zeros((obses.shape[0],))
    #         rewards_list = []
    #         for i in range(30):
    #             obses, rewards = self.model.rollout_out(actions)
    #             processed_rewards = self.preprocessor.tf_process_rewards(rewards)
    #             # reward_sum += self.tf.pow(self.args.gamma, i) * processed_rewards
    #             rewards_list.append(processed_rewards)
    #
    #             processed_obses = self.preprocessor.tf_process_obses(obses)
    #             actions, _ = self.policy_with_value.compute_action(processed_obses)
    #         for i in range(len(rewards_list)):
    #             reward_sum += self.tf.pow(self.args.gamma, i) * rewards_list[i]
    #         Qs = self.policy_with_value.compute_Qs(processed_obses, actions)[0][:, 0]
    #         target = self.tf.stop_gradient(reward_sum + self.tf.pow(self.args.gamma, 30) * Qs)
    #         q_loss = self.tf.reduce_mean(self.tf.square(target - q_pred))
    #
    #     q_gradient = tape.gradient(q_loss, self.policy_with_value.models[0].trainable_weights)
    #     return q_gradient, q_loss

    # def model_based_policy_forward_and_backward(self, mb_obs):
    #     with self.tf.GradientTape() as tape:
    #         obses = mb_obs
    #         self.model.reset(obses)
    #         reward_sum = self.tf.zeros((obses.shape[0],))
    #         rewards_list = []
    #         for i in range(30):
    #             processed_obses = self.preprocessor.tf_process_obses(obses)
    #             actions, _ = self.policy_with_value.compute_action(processed_obses)
    #             obses, rewards = self.model.rollout_out(actions)
    #             processed_rewards = self.preprocessor.tf_process_rewards(rewards)
    #             rewards_list.append(processed_rewards)
    #             # reward_sum += self.tf.pow(self.args.gamma, i) * processed_rewards
    #         for i in range(len(rewards_list)):
    #             reward_sum += self.tf.pow(self.args.gamma, i) * rewards_list[i]
    #         processed_obses = self.preprocessor.tf_process_obses(obses)
    #         actions, _ = self.policy_with_value.compute_action(processed_obses)
    #         Qs = self.policy_with_value.compute_Qs(processed_obses, actions)[0][:, 0]
    #         target = reward_sum + self.tf.pow(self.args.gamma, 30) * Qs
    #         policy_loss = -self.tf.reduce_mean(target)
    #
    #     policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)
    #     return policy_gradient, policy_loss

    @tf.function
    def model_based_policy_forward_and_backward(self, mb_obs):
        with self.tf.GradientTape() as tape:
            for i in range(self.forward_step):
                if i == 0:
                    self.obs_forward[i] = self.preprocessor.tf_process_obses(mb_obs)
                    self.action_forward[i], _ = self.policy_with_value.compute_action(self.obs_forward[i])
                    obs_next = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
                    self.obs_forward[i + 1] = self.preprocessor.tf_process_obses(obs_next)
                    self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])
                else:
                    self.action_forward[i], _ = self.policy_with_value.compute_action(self.obs_forward[i])
                    obs_next = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
                    self.obs_forward[i + 1] = self.preprocessor.tf_process_obses(obs_next)
                    self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])

            action_next, _ = self.policy_with_value.compute_action(self.obs_forward[-1])
            q_next = self.policy_with_value.compute_Qs(self.obs_forward[-1], action_next)[0][:, 0]
            target = self.tf.zeros_like(q_next)
            for i in range(self.forward_step):
                target += self.tf.pow(self.args.gamma, i) * self.reward_forward[i]
            target += self.tf.pow(self.args.gamma, self.forward_step) * q_next
            policy_loss = -self.tf.reduce_mean(target)

        policy_gradient = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)
        return policy_gradient, policy_loss

    @tf.function
    def model_based_q_forward_and_backward(self, mb_obs, mb_action):
        with self.tf.GradientTape() as tape:
            for i in range(self.forward_step):
                if i == 0:
                    self.obs_forward[i] = self.preprocessor.tf_process_obses(mb_obs)
                    self.action_forward[i] = mb_action
                    obs_next = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
                    self.obs_forward[i+1] = self.preprocessor.tf_process_obses(obs_next)
                    self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])
                else:
                    self.action_forward[i], _ = self.policy_with_value.compute_action(self.obs_forward[i])
                    obs_next = self.model.prediction(self.obs_forward[i], self.action_forward[i], 40., 1)
                    self.obs_forward[i + 1] = self.preprocessor.tf_process_obses(obs_next)
                    self.reward_forward[i] = self.model.compute_rewards(self.obs_forward[i], self.action_forward[i])

            action_next, _ = self.policy_with_value.compute_action(self.obs_forward[-1])
            q_next = self.policy_with_value.compute_Qs(self.obs_forward[-1], action_next)[0][:, 0]
            target = self.tf.zeros_like(q_next)
            for i in range(self.forward_step):
                target += self.tf.pow(self.args.gamma, i) * self.reward_forward[i]
            target += self.tf.pow(self.args.gamma, self.forward_step) * q_next
            q_pred = self.policy_with_value.compute_Qs(self.preprocessor.process_obs(mb_obs),
                                                       mb_action)[0][:, 0]
            q_loss = self.tf.reduce_mean(self.tf.square(self.tf.stop_gradient(target)-q_pred))

        q_gradient = tape.gradient(q_loss, self.policy_with_value.models[0].trainable_weights)
        return q_gradient, q_loss

    def export_graph(self, writer):
        start_idx, end_idx = 0, self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        self.tf.summary.trace_on(graph=True, profiler=False)
        if self.args.model_based:
            self.model_based_q_forward_and_backward(mb_obs, mb_actions)
        else:
            self.q_forward_and_backward(mb_obs, mb_actions, mb_tdlambda_returns)
        with writer.as_default():
            self.tf.summary.trace_export(name="q_forward_and_backward", step=0)

        self.tf.summary.trace_on(graph=True, profiler=False)
        if self.args.model_based:
            self.model_based_policy_forward_and_backward(mb_obs)
        else:
            self.policy_forward_and_backward(mb_obs)
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
        mb_obs = self.batch_data['batch_obs'][start_idx: end_idx]
        mb_tdlambda_returns = self.batch_data['batch_tdlambda_returns'][start_idx: end_idx]
        mb_actions = self.batch_data['batch_actions'][start_idx: end_idx]
        # judge_is_nan([mb_obs])
        # judge_is_nan([processed_mb_obs])
        # judge_is_nan([mb_advs])
        # judge_is_nan([mb_tdlambda_returns])
        # judge_is_nan([mb_actions])
        # judge_is_nan([mb_neglogps])

        # print(self.preprocessor.get_params())

        with self.q_gradient_timer:
            if self.args.model_based:
                q_gradient, q_loss = self.model_based_q_forward_and_backward(mb_obs, mb_actions)
                w_q_list = [1.]
            else:
                model_targets, w_q_list, q_gradient, q_loss = self.q_forward_and_backward(mb_obs, mb_actions,
                                                                                          mb_tdlambda_returns)
            q_gradient, q_gradient_norm = self.tf.clip_by_global_norm(q_gradient, self.args.gradient_clip_norm)

        with self.policy_gradient_timer:
            self.policy_for_rollout.set_weights(self.policy_with_value.get_weights())
            if self.args.model_based:
                final_policy_gradient, value_mean = self.model_based_policy_forward_and_backward(mb_obs)
            else:
                model_returns, minus_reduced_model_returns, jaco, value_mean = self.policy_forward_and_backward(mb_obs)
        # print(jaco, type(jaco))
        # print(model_returns, type(model_returns))
        # judge_is_nan([model_returns])
        # judge_is_nan(jaco)

        policy_gradient_list = []
        heuristic_bias_list = []
        var_list = []
        # final_policy_gradient = []
        w_heur_bias_list = []
        w_var_list = []
        w_list = []
        #
        # for rollout_index in range(len(self.num_rollout_list_for_policy_update)):
        #     jaco_for_this_rollout = list(map(lambda x: x[rollout_index * self.reduced_num_minibatch:
        #                                                  (rollout_index + 1) * self.reduced_num_minibatch], jaco))
        #
        #     gradient_std = []
        #     gradient_mean = []
        #     var = 0.
        #     for x in jaco_for_this_rollout:
        #         gradient_std.append(self.tf.math.reduce_std(x, 0))
        #         gradient_mean.append(self.tf.reduce_mean(x, 0))
        #         var += self.tf.reduce_mean(self.tf.square(gradient_std[-1])).numpy()
        #     heuristic_bias = self.tf.reduce_mean(
        #         self.tf.square(model_returns[rollout_index * self.args.mini_batch_size:
        #                                      (rollout_index + 1) * self.args.mini_batch_size]
        #                        - mb_tdlambda_returns)).numpy()
        #
        #     # judge_is_nan(gradient_mean)
        #
        #     policy_gradient_list.append(gradient_mean)
        #     heuristic_bias_list.append(heuristic_bias)
        #     var_list.append(var)
        #     # judge_is_nan(var_list)
        #     # judge_is_nan(heuristic_bias_list)
        #
        # epsilon = 1e-8
        # heuristic_bias_inverse_sum = self.tf.reduce_sum(
        #     list(map(lambda x: 1. / (x + epsilon), heuristic_bias_list))).numpy()
        # var_inverse_sum = self.tf.reduce_sum(list(map(lambda x: 1. / (x + epsilon), var_list))).numpy()
        #
        # w_heur_bias_list = list(
        #     map(lambda x: (1. / (x + epsilon)) / heuristic_bias_inverse_sum, heuristic_bias_list))
        # w_var_list = list(map(lambda x: (1. / (x + epsilon)) / var_inverse_sum, var_list))
        #
        # w_list = list(map(lambda x, y: (x + y) / 2., w_heur_bias_list, w_var_list))
        #
        # # judge_is_nan(w_list)
        #
        # for i in range(len(policy_gradient_list[0])):
        #     tmp = 0
        #     for j in range(len(policy_gradient_list)):
        #         # judge_is_nan(policy_gradient_list[j])
        #         tmp += w_list[j] * policy_gradient_list[j][i]
        #     final_policy_gradient.append(tmp)
        #
        # judge_is_nan(q_gradient)
        # judge_is_nan(final_policy_gradient)

        final_policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(final_policy_gradient,
                                                                                  self.args.gradient_clip_norm)

        self.stats.update(dict(num_traj_rollout=self.M,
                               num_rollout_list=self.num_rollout_list_for_policy_update,
                               q_timer=self.q_gradient_timer.mean,
                               pg_time=self.policy_gradient_timer.mean,
                               q_loss=q_loss.numpy(),
                               value_mean=value_mean.numpy(),
                               w_q_list=[],  # list(map(lambda x: x.numpy(), w_q_list)),
                               var_list=var_list,
                               heuristic_bias_list=heuristic_bias_list,
                               w_var_list=w_var_list,
                               w_heur_bias_list=w_heur_bias_list,
                               w_list=w_list,
                               q_gradient_norm=q_gradient_norm.numpy(),
                               policy_gradient_norm=policy_gradient_norm.numpy()))

        gradient_tensor = q_gradient + final_policy_gradient  # q_gradient + final_policy_gradient
        return np.array(list(map(lambda x: x.numpy(), gradient_tensor)))


if __name__ == '__main__':
    pass
