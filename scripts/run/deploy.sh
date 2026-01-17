# export SWANLAB_API_KEY=hlDF5JsRQoFLOA6Kd9tbX
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import site; print(site.getsitepackages()[0])')/nvidia/npp/lib

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

config=$1
ckpt_path=$2
ARGS=${@:3}

config="${config#configs/}" # delete prefix configs/
config="${config#task/}" # delete prefix task/
config="${config%.yaml}" # delete suffix .yaml

python scripts/deploy.py task=$config ckpt_path=$ckpt_path $ARGS

# bash scripts/run/deploy.sh real/r1lite_g0plus_finetune_deploy xxx.pt