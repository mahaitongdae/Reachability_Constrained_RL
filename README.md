# Mixed Policy Gradient (MPG)
The repository MPG is originated from the paper "Mixed policy gradient" (see [here](https://arxiv.org/abs/2102.11513) 
for details), which contains a cluster of high-quality implementations of deep reinforcement learning algorithms,
including the proposed MPG and other baseline algorithms, namely n-step Approximate Dynamic Programming (n-step ADP), 
n-step Deterministic Policy Gradient (n-step DPG), Twin Delayed Deep Deterministic policy gradient (TD3), 
and Soft Actor-Critic (SAC). In addition, we also implemented the widely used on-policy algorithms such as
 Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO) (see branch ```MPG_on_policy```).

The implementation is fairly thin and primarily optimized for our own development purposes. It is designed with TensorFlow 2 and Ray
to realize a high-throughput asynchronous learning architecture, which modularizes the process of sampling, storing, learning, 
evaluating and testing with clear interfaces, organizes each of them in parallel, as shown below. This architecture can help to
scale to hundreds of cpu cores to largely enhance the sampling and update throughput. Besides, with the general design, 
most of the gradient-based reinforcement learning algorithms can be incorporated.

# Get started
Run the script files under the train_scripts folder, in which you can choose an algorithm and modify its related 
parameters. Then, enjoy it. :)

# References
The algorithms are based on the following papers:

Guan, Y., Duan, J., Li, S. E., Li, J., Chen, J., & Cheng, B. (2021). Mixed Policy Gradient. arXiv preprint arXiv:2102.11513.

Silver, David, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. "Deterministic policy gradient algorithms." In International conference on machine learning, pp. 387-395. PMLR, 2014.

Fujimoto, Scott, Herke Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." In International Conference on Machine Learning, pp. 1587-1596. PMLR, 2018.

Haarnoja, Tuomas, Aurick Zhou, Pieter Abbeel, and Sergey Levine. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." In International Conference on Machine Learning, pp. 1861-1870. PMLR, 2018.

Schulman, John, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. "Trust region policy optimization." In International conference on machine learning, pp. 1889-1897. PMLR, 2015.

Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

If MPG helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```
@article{guan2021mixed,
  title={Mixed Policy Gradient},
  author={Guan, Yang and Duan, Jingliang and Li, Shengbo Eben and Li, Jie and Chen, Jianyu and Cheng, Bo},
  journal={arXiv preprint arXiv:2102.11513},
  year={2021}
}
```
