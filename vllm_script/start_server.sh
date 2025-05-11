CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg \
    --served-model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --trust-remote-code \
    --task generate \
    --gpu-memory-utilization 0.65 \
    --port 23333 \
    --quantization modelopt
    # --data-parallel-size 2