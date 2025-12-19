export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_NODES=1
export CURR_NODE_RANK=0
export WORLD_SIZE=4

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc-per-node=${WORLD_SIZE} \
  --node-rank=${CURR_NODE_RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:12333 \
  train.py --config configs/urbannav_debug.yaml
  