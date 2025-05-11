for batch_size in 1 2 4 8 16 32 64 128 256; do
    for input_len in 512 8192 32768; do
        for output_len in 32 256; do
            max_concurrency=$((batch_size * 1))
            echo "Running with batch-size=${batch_size}, input-len=${input_len}, output-len=${output_len}"
            python vllm/benchmarks/benchmark_serving.py \
                --backend vllm --dataset-name random \
                --model Qwen/Qwen2.5-VL-3B-Instruct \
                --random-input-len ${input_len} \
                --random-output-len ${output_len} \
                --random-range-ratio 0.99 \
                --port 23333 \
                --num-prompts 20 \
                --max-concurrency ${max_concurrency} \
                --save-result \
                --append-result \
                --result-filename throughput_server_noDP.jsonl
        done
    done
done