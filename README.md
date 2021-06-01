# Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety

<div align=center>
<img src="utils/walker.gif" width = 25%/>
<img src="utils/safexp.gif" width = 25%/>
</div>

This repository is the official implementation of Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety. 
The code base of this implementation is the [Parallel Asynchronous Buffer-Actor-Learner (PABAL) architecture](https://github.com/idthanm/mpg),
which includes implementations of most common RL algorithms with the state-of-the-art high efficiency.
If you are interested in or want to contribute to PABAL, you can contact me or the [original creator](https://github.com/idthanm).

## Requirements

**Important information for installing the requirements:**
1. We test it successfully only on **Python 3.6**, and higher python version causes error with Safety Gym and TensorFlow 2.x. 
2. Make sure you have installed [MuJoCo and mujoco-py](https://github.com/openai/mujoco-py) properly.
3. Safety Gym and TensorFlow 2.x have conflict in numpy version. We test on numpy 1.17.5. If it runs with errors, pls check the numpy version.

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train_scripts4fsac.py --env_id Safexp-PointButton1-v0 --seed 0
```


## Evaluation

To test and evaluate trained policies, run:

```test
python train_scripts4fsac.py --mode testing --test_dir <your_log_dir> --test_iter_list [3000000]
```



## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with me before making a change.


