# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang_router.launch_server \
#     --model-path Qwen/Qwen2.5-VL-3B-Instruct \
#     --dp 4 \
#     --chat-template qwen2-vl \
#     --mem-fraction-static 0.9 \
#     --port 30000

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang_router.launch_server \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --dp 4 \
    --chat-template qwen2-vl \
    --mem-fraction-static 0.9 \
    --port 30000