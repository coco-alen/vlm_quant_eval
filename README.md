
# 2025/04/22

## Install

```bash
pip install -r requirements.txt
```

Image/Video tasks evaluation codebase: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main)

## Throughput Test

based on sglang

### fp16/bf16

1、start sglang server

```bash
python -m sglang_router.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --dp 4 --chat-template qwen2-vl
```

2、test it

```bash
python -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 30000 --random-output-len 128 --random-range-ratio 1 --profile --apply-chat-template --num-prompts 200 --max-concurrency 1
```


Or - directly test the model

```bash
python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-VL-3B-Instruct --batch 1 --input-len 30000 --output-len 128 --chat-template qwen2-vl --dp 4
```

