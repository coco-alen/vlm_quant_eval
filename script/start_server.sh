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

CUDA_VISIBLE_DEVICES=4,5 python -m sglang_router.launch_server \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --chat-template qwen2-vl \
    --data-parallel-size 2 \
    --router-balance-rel-threshold 1.1 \
    --router-balance-abs-threshold 1 \
    --context-length 35000 \
    --port 23333 \
    --max-running-requests 256 \
    --quantization w8a8_fp8 \
    --mem-fraction-static 0.66 \
    --chunked-prefill-size 4096