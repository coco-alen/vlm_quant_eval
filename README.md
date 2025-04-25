
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

#### 1. start sglang server

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang_router.launch_server \
    --model-path Qwen/Qwen2.5-VL-3B-Instruct \
    --chat-template qwen2-vl \
    --mem-fraction-static 0.9 \
    --max-running-requests 16 \
    --port 23333 \
    --load-balance-method shortest_queue \
    --data-parallel-size 4
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

#### 2. bench the tasks

[Task list](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewa8sGZrY)

```bash
cd VLMEvalKit

python run.py \
    --data MMMU_DEV_VAL MEGABench_core_16frame DocVQA_[VAL/TEST] InfoVQA_[VAL/TEST] VideoMMLU_CAP_16frame VideoMMLU_QA_16frame Video-MME_8frame MathVista_MINI\
    --model lmdeploy 
    --api-nproc 4 \
    --work-dir './outputs'
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



## Bug Fix

#### 1. w8a8_int8 in sglang

> sglang/srt/layers/quantization/w8a8_int8.py    Line 111

change *apply* func to:

```python
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        x_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], layer.weight.shape[1]]
        x_q, x_scale = per_token_quant_int8(x_2d)
        output = int8_scaled_mm(
            x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias
        )
        return output.view(*output_shape)
```

**Explain**: When using the API to call the service, there is a batch size dimension. This bug will not be triggered in fp8 because a different implementation path is used. A similar implementation exists for int8 (*int8_utils.py*), but it is not used.



#### 2. Disable ViT Mlp quantization

In the MLP structure of ViT of Qwen2.5-VL, the shape is *3420*, which is not divisible by 8, which causes the kernel to report an error. Therefore, we need to manually turn off the quantization of the MLP structure.

> sglang/srt/models/qwen2_5_vl.py    Line 77  in  *class Qwen2_5_VLMLP(nn.Module):*

```python
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            # quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            # quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            # quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act = ACT2FN[hidden_act]
```

Or you can also disable the quant of the entire ViT, just comment out the *quant_config* of *Qwen2_5_VisionTransformer* on line 460

