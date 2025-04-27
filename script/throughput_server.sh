
for batch_size in 16 32 64 128; do
    for input_len in 512 8192 32768; do
        for output_len in 32 256; do
            max_concurrency=$((batch_size * 2))
            echo "Running with batch-size=${batch_size}, input-len=${input_len}, output-len=${output_len}"
            python -m sglang.bench_serving --backend sglang --dataset-name random \
                --random-input-len ${input_len} \
                --random-output-len ${output_len} \
                --random-range-ratio 1 \
                --port 23333 \
                --profile \
                --apply-chat-template \
                --warmup-requests 5 \
                --num-prompts 20 \
                --max-concurrency ${max_concurrency}
        done
    done
done