import modelopt.torch.quantization as mtq
from transformers import AutoProcessor, AutoModelForImageTextToText
import transformers
import torch
from modelopt.torch.export import export_hf_checkpoint

model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processer = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print(model)

config = mtq.NVFP4_DEFAULT_CFG
model = mtq.quantize(model, config)
print(model)

# with torch.inference_mode():
#     export_hf_checkpoint(
#         model,  # The quantized model.
#         "/mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg",  # The directory where the exported files will be stored.
#     )

model.save_pretrained("/mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg")
processer.save_pretrained("/mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg")
