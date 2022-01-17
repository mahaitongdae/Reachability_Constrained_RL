export PYTHONPATH=/home/yudongjie/Projects/Reachability_Constrained_RL:$PYTHONPATH

python ./train_script.py

python ./train_script4saclag.py

# python ./train_script4fsac.py

python ./train_script4rew_shaping.py

python ./train_script4cbf.py

python ./train_script4energy.py --random_seed 123