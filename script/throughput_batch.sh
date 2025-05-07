# for quantize in awq; do
#     for batch_size in 1 2 4 8; do
#         for input_len in 512 8192 32768; do
#             for output_len in 32 128 256; do
#                 echo "Running with batch-size=${batch_size}, input-len=${input_len}, output-len=${output_len}, quant=${quantize}"
#                 CUDA_VISIBLE_DEVICES=4 python -m sglang.bench_one_batch \
#                     --model-path Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
#                     --dp 4 \
#                     --chat-template qwen2-vl \
#                     --dtype float16 \
#                     --mem-fraction-static 0.9 \
#                     --quantization ${quantize} \
#                     --port 30000 \
#                     --run-name "${quantize}_bs${batch_size}_in${input_len}_out${output_len}" \
#                     --batch-size ${batch_size} \
#                     --input-len ${input_len} \
#                     --output-len ${output_len} \
#                     --result-filename "${quantize}_in${input_len}_out${output_len}"
#             done
#         done
#     done
# done

# for quantize in w8a8_int8 w8a8_fp8 modelopt_fp4; do


export SGLANG_TORCH_PROFILER_DIR=/mnt/raid0/yipin/quant_eval
CUDA_VISIBLE_DEVICES=0 python -m sglang.bench_one_batch \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --chat-template qwen2-vl \
    --dtype float16 \
    --mem-fraction-static 0.3 \
    --port 30000 \
    --run-name "test" \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 32 \
    --result-filename "test" \
    --profile
