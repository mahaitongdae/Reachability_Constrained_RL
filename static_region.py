from policy import Policy4Lagrange
import os
from evaluator import Evaluator
import gym
from utils.em_brake_4test import EmergencyBraking
import numpy as np
from matplotlib.colors import ListedColormap
from gym.envs.user_defined.EmerBrake.models import EmBrakeModel


def model_rollout_for_update(self, start_obses, ite):
    start_obses = self.tf.tile(start_obses, [self.M, 1])
    self.model.reset(start_obses)
    rewards_sum = self.tf.zeros((start_obses.shape[0],))
    obses = start_obses
    constraints_list = []
    for step in range(self.num_rollout_list_for_policy_update[0]):
        processed_obses = self.preprocessor.tf_process_obses(obses)
        actions, _ = self.policy_with_value.compute_action(processed_obses)
        obses, rewards, constraints = self.model.rollout_out(actions)
        constraints_clip = self.tf.clip_by_value(constraints, CONSTRAINTS_CLIP_MINUS, 100)
        constraints_list.append(self.tf.expand_dims(constraints_clip, axis=1))

        rewards_sum += self.preprocessor.tf_process_rewards(rewards)
    constraints_all = self.tf.concat(constraints_list, 1)
    processed_start_obses = self.preprocessor.tf_process_obses(start_obses)
    mu_all = self.policy_with_value.compute_mu(processed_start_obses)
    cs_sum = self.tf.reduce_sum(self.tf.multiply(mu_all, self.tf.stop_gradient(constraints_all)), 1)
    punish_terms_sum = self.tf.reduce_sum(self.tf.multiply(self.tf.stop_gradient(mu_all), constraints_all), 1)

    obj_loss = -self.tf.reduce_mean(rewards_sum)
    punish_terms = self.tf.reduce_mean(punish_terms_sum)
    pg_loss = obj_loss + punish_terms
    cs_loss = -self.tf.reduce_mean(cs_sum)
    constraints = self.tf.reduce_mean(constraints_all)

    return obj_loss, punish_terms, cs_loss, pg_loss, constraints



def static_region(test_dir, iteration):
    import json
    import argparse
    import datetime
    from train_script import built_LMAMPC_parser
    from policy import Policy4Lagrange
    # args = built_LMAMPC_parser()
    params = json.loads(open(test_dir + '/config.json').read())
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_log_dir = params['log_dir'] + '/tester/test-region-{}'.format(time_now)
    params.update(dict(mode='testing',
                       test_dir=test_dir,
                       test_log_dir=test_log_dir,))
    parser = argparse.ArgumentParser()
    for key, val in params.items():
        parser.add_argument("-" + key, default=val)
    args =  parser.parse_args()
    evaluator = Evaluator(Policy4Lagrange, args.env_id, args)
    evaluator.load_weights(os.path.join(test_dir, 'models'), iteration)
    model = EmBrakeModel()

    # generate batch obses
    d = np.linspace(0, 10, 100)
    v = np.linspace(0, 10, 100)
    # cmaplist = ['springgreen'] * 3 + ['crimson'] * 87
    # cmap1 = ListedColormap(cmaplist)
    D, V = np.meshgrid(d, v)
    flatten_d = np.reshape(D, [-1, ])
    flatten_v = np.reshape(V, [-1, ])
    init_obses = np.stack([flatten_d, flatten_v], 1)

    # define rollout
    def reduced_model_rollout_for_update(obses):
        model.reset(obses)
        constraints_list = []
        for step in range(args.num_rollout_list_for_policy_update[0]):
            processed_obses = evaluator.preprocessor.tf_process_obses(obses)
            actions, _ = evaluator.policy_with_value.compute_action(processed_obses)
            obses, rewards, constraints = model.rollout_out(actions)
            constraints_list.append(constraints)
        flattern_cstr = evaluator.tf.stack(constraints_list, 1).numpy()
        return flattern_cstr
    flatten_cstr = reduced_model_rollout_for_update(init_obses)

    preprocess_obs = evaluator.preprocessor.np_process_obses(init_obses)
    flatten_mu = evaluator.policy_with_value.compute_mu(preprocess_obs).numpy()

    flatten_cs = np.multiply(flatten_cstr, flatten_mu)

    def plot_region(data, name):
        for k in range(data.shape[1]):
            data_k = data[:, k]
            data_reshape = data_k.reshape(D.shape)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            plt.figure()
            plt.grid()
            x = np.linspace(0, 10)
            t = np.sqrt(2 * 5 * x)
            ct = plt.contour(D, V, data_reshape, 50, cmap='rainbow')
            plt.colorbar(ct)
            plt.plot(x, t, linestyle='--', color='red')
            name_2d = name + '_' + str(k) + '_2d.jpg'
            plt.savefig(os.path.join(evaluator.log_dir, name_2d))

            figure = plt.figure()
            ax = Axes3D(figure)
            ax.plot_surface(D, V, data_reshape, rstride=1, cstride=1, cmap='rainbow')
            # plt.show()
            name_3d = name + '_' + str(k) + '_3d.jpg'
            plt.savefig(os.path.join(evaluator.log_dir, name_3d))

    plot_region(flatten_cs, 'cs')



if __name__ == '__main__':
    static_region('./results/toyota3lane/LMAMPC-v2-2021-06-24-14-10-01', 200000)