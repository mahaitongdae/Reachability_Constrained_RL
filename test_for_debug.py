import ray
import numpy as np
from utils.memory import ray_get_and_free
from utils.task_pool import TaskPool
import tensorflow as tf



class Sample_numpy(object):
    def __init__(self):
        pass

    def sample(self):
        samples = np.random.random([1000, 1000])
        return samples
#
#
def test_remote():
    import sys
    ray.init(redis_max_memory=100 * 1024 * 1024, object_store_memory=100 * 1024 * 1024)
    sps = [ray.remote(Sample_numpy).remote() for _ in range(1)]
    sample_tasks = TaskPool()
    for sp in sps:
        sample_tasks.add(sp, sp.sample.remote())
    samples = None
    for _ in range(1000000000):
        for sp, objID in list(sample_tasks.completed(blocking_wait=True)):
            samples = ray.get(objID)
            sample_tasks.add(sp, sp.sample.remote())


def test_remote2():
    ray.init(redis_max_memory=100 * 1024 * 1024, object_store_memory=1000 * 1024 * 1024)
    sampler = ray.remote(Sample_numpy).remote()
    for _ in range(1000000000):
        objID = sampler.sample.remote()
        samples = ray.get(objID)
        # ray.internal.free([objID])

        # hxx = hpy()
        # heap = hxx.heap()
        # print(heap.byrcs)


# def test_local():
#     a = Sample_numpy()
#     for _ in range(1000000000):
#         sample = a.sample()
#
#
# def test_get_size():
#     import sys
#     samples = np.random.random([1000, 1000])
#     print(sys.getsizeof(samples)/(1024*1024))
#
#
# def test_memory_profiler():
#     c = []
#     a = [1, 2, 3] * (2 ** 20)
#     b = [1] * (2 ** 20)
#     c.extend(a)
#     c.extend(b)
#     del b
#     del c


def test_jacobian():
    variable = tf.Variable(1.0)
    inputs = (
        tf.constant(tf.random.uniform((1, 4))),
        tf.constant(tf.random.uniform((1, 3))),
    )
    print(inputs)

    with tf.GradientTape(persistent=True) as tape:
        outputs = variable * tf.pow(tf.concat(inputs, axis=-1), 2.0)

    print(outputs)

    jacobians_1 = tape.jacobian(
        outputs,
        variable,
        experimental_use_pfor=True,
    )
    print(jacobians_1)
    print("tape.jacobians(..., experimental_use_pfor=True) works!")

    try:
        jacobians_2 = tape.jacobian(
            outputs,
            variable,
            experimental_use_pfor=False,
        )
        print(jacobians_2)
        print("tape.jacobians(..., experimental_use_pfor=False) works!")
    except TypeError:
        print("tape.jacobians(..., experimental_use_pfor=False) doesn't work!")
        raise

def test_jacobian2():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(2,)),
    ])
    print(model.trainable_variables)
    inputs = tf.Variable([[1, 2]], dtype=tf.float32)

    with tf.GradientTape() as gtape:
        outputs = model(inputs)
    print(outputs)
    jaco = gtape.jacobian(outputs, model.trainable_variables)
    print(jaco)

def test_logger():
    import logging
    import os
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.error('Watch out!')

    logger.warning('2222')
    logger.info('sdsdf')

def test_tf22():
    from tensorflow.keras.optimizers import Adam
    a = Adam()
    print(a)

def tape_gra():
    from train_script import built_mixedpg_parser
    import gym
    from policy import PolicyWithQs
    import numpy as np
    import tensorflow as tf
    args = built_mixedpg_parser()
    args.obs_dim = 3
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)

    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
    inp = tf.Variable(3.)

    with tf.GradientTape(persistent=True) as tape:
        acts, _ = policy_with_value.compute_action(obses)
        Qs = policy_with_value.compute_Qs(obses, acts)[0]
        c = np.array([1.,2.,3.])*inp
        out = []
        for ci in c:
            out.append(ci)
        a = Qs[0][0]
        c0=c[0]
        c1=c[1]

    gradient = tape.jacobian(c, inp)
    print(gradient)

def test_gym():
    import gym
    env = gym.make('PathTracking-v0')
    obs = env.reset()
    print(obs)
    action = np.array([0, 10000])
    obs = env.step(action)
    print(obs)
    a = 1


def test_ray_init():
    ray.init()
    ray.init()


if __name__ == '__main__':
    test_ray_init()

