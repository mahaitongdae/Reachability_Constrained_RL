# Reachability Constrained Reinforcement Learning

This repository is the official implementation of the quadrotor experiments in [*Reachability Constrained Reinforcement Learning*](https://arxiv.org/abs/2205.07536).
The code base of this implementation is the [Parallel Asynchronous Buffer-Actor-Learner (PABAL) architecture](https://github.com/idthanm/mpg),
which includes implementations of most common RL algorithms with the state-of-the-art training efficiency.
If you are interested in or want to contribute to PABAL, you can contact me or the [original creator](https://github.com/idthanm).

## Requirements
**[Safe-control-gym](https://github.com/ManUtdMoon/safe-control-gym) is needed before running!**

To install other requirements:

```setup
$ pip install -U ray
$ pip install tensorflow==2.5.0
$ pip install tensorflow_probability==0.13.0
$ pip install seaborn matplotlib
```

## Training
To train the algorithm(s) in the paper, run these commands or directly run `sh bash.sh` in `/train_scripts/`:
```train
$ export PYTHONPATH=/your/path/to/Reachability_Constrained_RL/:$PYTHONPATH
$ cd ./train_scripts/
$ python train_scripts.py                # RCRL (RAC)
$ python train_scripts4saclag.py         # SAC-Lagrangian
$ python train_scripts4rew_shaping.py    # SAC-Reward Shaping
$ python train_scripts4cbf.py            # SAC-CBF
$ python train_scripts4energy.py         # SAC-SI
```
**Each script needs about 28 CPU threads and 9 hours to run.** Thus, it will take much time.


### Training supervision
Results can be seen with tensorboard:
```
$ cd ./results/
$ tensorboard --logdir=. --bindall
```

## Evaluation
To test and evaluate trained policies, run:

```test
python train_scripts4<alg_name>.py --mode testing --test_dir <your_log_dir> --test_iter_list <iter_nums>
```
and the results will be recored in `/results/quadrotor/<ALGO_NAME>/<EXP_TIME>/logs/tester`.

### Results visualization
`visualize_scripts/visualize_quadrotor_trajectory.py` and `visualize_scripts/visualize_region_quadrotor.py` can be used to visualize the trajectory and the feasible set, respectively. All you need to do is to paste the directory of the experiment run into the main function.

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with me before making a change.

## If you find our paper/code helpful, welcome to cite:
```
@InProceedings{yuandma2022rcrl,
  title = 	 {Reachability Constrained Reinforcement Learning},
  author =       {Yu, Dongjie and Ma, Haitong and Li, Shengbo and Chen, Jianyu},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {25636--25655},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/yu22d/yu22d.pdf},
  url = 	 {https://proceedings.mlr.press/v162/yu22d.html},
}
```
