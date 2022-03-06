#!/bin/sh

PARENT_DIR=$(cd $(dirname $0);cd ..; pwd)
export PYTHONPATH=$PYTHONPATH:$PARENT_DIR

# python ./train_script.py

python ./train_script4saclag.py

# python ./train_script4fsac.py

# python ./train_script4rew_shaping.py

# python ./train_script4cbf.py

# python ./train_script4energy.py