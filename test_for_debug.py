import ray
import numpy as np
from utils.memory import ray_get_and_free
from utils.task_pool import TaskPool



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


if __name__ == '__main__':
    test_remote()

