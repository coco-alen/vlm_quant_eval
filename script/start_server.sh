# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang_router.launch_server \
#     --model-path Qwen/Qwen2.5-VL-3B-Instruct \
#     --dp 4 \
#     --chat-template qwen2-vl \
#     --mem-fraction-static 0.9 \
#     --port 30000

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
#     --model-path Qwen/Qwen2.5-VL-3B-Instruct \
#     --chat-template qwen2-vl \
#     --mem-fraction-static 0.65 \
#     --quantization w8a8_fp8 \
#     --max-running-requests 4 \
#     --port 23333 \
#     --api-key miniphant
export SGLANG_TORCH_PROFILER_DIR=/mnt/raid0/yipin/quant_eval

# CUDA_VISIBLE_DEVICES=0,1 python -m sglang_router.launch_server \
#     --model-path /mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg \
#     --chat-template qwen2-vl \
#     --data-parallel-size 2 \
#     --router-balance-rel-threshold 1.1 \
#     --router-balance-abs-threshold 1 \
#     --context-length 50000 \
#     --quantization modelopt_fp4 \
#     --port 23333 \
#     --max-running-requests 256 \
#     --mem-fraction-static 0.5 \
#     --chunked-prefill-size 4096

CUDA_VISIBLE_DEVICES=5 python -m sglang.launch_server \
    --model-path /mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg \
    --chat-template qwen2-vl \
    --context-length 50000 \
    --quantization modelopt_fp4 \
    --port 23333 \
    --max-running-requests 256 \
    --mem-fraction-static 0.5 \
    --chunked-prefill-size 4096