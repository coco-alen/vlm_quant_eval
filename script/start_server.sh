
export SGLANG_TORCH_PROFILER_DIR=/mnt/raid0/yipin/quant_eval

# CUDA_VISIBLE_DEVICES=4,5 python -m sglang_router.launch_server \
#     --model-path /mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg \
#     --chat-template qwen2-vl \
#     --data-parallel-size 2 \
#     --router-balance-rel-threshold 1.1 \
#     --router-balance-abs-threshold 1 \
#     --quantization modelopt_fp4 \
#     --context-length 50000 \
#     --port 23333 \
#     --max-running-requests 256 \
#     --mem-fraction-static 0.5 \
#     --chunked-prefill-size 4096

CUDA_VISIBLE_DEVICES=4 python -m sglang.launch_server \
    --model-path /mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg \
    --chat-template qwen2-vl \
    --context-length 50000 \
    --port 23333 \
    --quantization modelopt_fp4 \
    --max-running-requests 256 \
    --mem-fraction-static 0.65 \
    --chunked-prefill-size 4096