
# 2025/04/22

## Install

```bash
pip install -r requirements.txt
```

install Evaluation codebase: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit/tree/main)

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```



## Throughput Bench

based on sglang

### fp16/bf16

1、start sglang server

```bash
python -m sglang_router.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --dp 4 --chat-template qwen2-vl
```

2、bench it

```bash
python -m sglang.bench_serving --backend sglang --dataset-name random --random-input-len 30000 --random-output-len 128 --random-range-ratio 1 --profile --apply-chat-template --num-prompts 200 --max-concurrency 1
```


Or - directly bench the model

```bash
python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-VL-3B-Instruct --batch 1 --input-len 30000 --output-len 128 --chat-template qwen2-vl --dp 4
```



## Task Bench

1、start sglang server

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang_router.launch_server \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --dp 4 \
    --chat-template qwen2-vl \
    --mem-fraction-static 0.9 \
    --port 23333
```

port 23333 is necessary. Or you can change the code in VLMEvalKit

> VLMEvalKit/vlmeval/config.py
>
> line 382

```python
    "lmdeploy": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
```

2、bench the tasks

[Task list](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY)

```bash
cd VLMEvalKit

export LOCAL_LLM="Qwen/Qwen2.5-VL-3B-Instruct"

python run.py \
    --data MMMU_DEV_VAL MEGABench_core_16frame DocVQA_[VAL/TEST] InfoVQA_[VAL/TEST] VideoMMLU_CAP_16frame VideoMMLU_QA_16frame Video-MME_8frame MathVista_MINI\
    --model lmdeploy 
    # --verbose


# MMMU_DEV_VAL 
# MEGABench_open_64frame MEGABench_open_16frame MEGABench_core_64frame MEGABench_core_16frame
# DocVQA_[VAL/TEST]
# InfoVQA_[VAL/TEST]
# VideoMMLU_CAP_16frame VideoMMLU_CAP_64frame VideoMMLU_QA_16frame VideoMMLU_QA_64frame
# Video-MME_8frame Video-MME_64frame Video-MME_8frame_subs Video-MME_1fps Video-MME_0.5fps Video-MME_0.5fps_subs
# MathVista_MINI
```

--verbose  :  print the answer of VLM