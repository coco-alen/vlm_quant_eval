# import os
# import torch
# import modelopt.torch.quantization as mtq

# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from transformers import AutoProcessor, AutoModelForImageTextToText

# from datasets import load_dataset
# from modelopt.torch.export import export_hf_checkpoint
# from qwen_vl_utils import process_vision_info

# # calibrate
# dataset = load_dataset("lmms-lab/MME" , split="test")
# # 设置 batch_size 和最大样本数
# batch_size = 4
# calib_size = 16 * batch_size
# export_dir = "/mnt/raid0/yipin/quant_eval/model/qwen2vl-3b-fp4-defaultCfg"
# os.makedirs(export_dir, exist_ok=True)
# # 取一部分数据用于校准
# dataset = dataset.select(range(calib_size))
# model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").to("cuda:4")
# processer = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


# def collate_fn(batch):
#     texts = [item['question'] for item in batch]
#     images = [item['image'].convert('RGB') for item in batch]  # 确保图像是 RGB 格式
#     return {"text": texts, "image": images}

# # 创建 DataLoader
# data_loader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     collate_fn=collate_fn,
# )

# def forward_loop(model):

#     for batch in tqdm(data_loader, desc="Calibrating"):
#         text_list = batch["text"]
#         image_list = batch["image"]
#         messages = []
#         for idx in range(len(text_list)):
#             messages.append(
#                 [{
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "image",
#                             "image": image_list[idx],
#                         },
#                         {"type": "text", "text": text_list[idx]},
#                     ],
#                 }]
#             )
#         text = [processer.apply_chat_template(
#                 msg, tokenize=False, add_generation_prompt=True
#             ) for msg in messages ]
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processer(
#             text=text,
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )

#         # 移动到 GPU
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}

#         # 前向传播
#         with torch.no_grad():
#             model(**inputs)

# config = mtq.NVFP4_DEFAULT_CFG
# # config = mtq.W4A8_AWQ_BETA_CFG
# config["quant_cfg"]['*visual*'] = {'enable': False}
# config["quant_cfg"]['*merger*'] = {'enable': False}
# print(config)
# model = mtq.quantize(model, config, forward_loop)
# print(model)

# with torch.inference_mode():
#     export_hf_checkpoint(
#         model,  # The quantized model.
#         export_dir = export_dir,  # The directory where the exported files will be stored.
#         save_modelopt_state=False,  # Whether to save the modelopt state.
#     )

# processer.save_pretrained(export_dir)


import os
import torch
import modelopt.torch.quantization as mtq

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint

# calibrate
dataset = load_dataset("cais/mmlu" , "all")['test']
# 设置 batch_size 和最大样本数
batch_size = 16
calib_size = 16 * batch_size
export_dir = "model/Qwen2.5-1.5B-Instruct"
os.makedirs(export_dir, exist_ok=True)
# 取一部分数据用于校准
dataset = dataset.select(range(calib_size))
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


def collate_fn(batch):
    texts = [item['question'] for item in batch]
    return {"text": texts}

# 创建 DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
)

def forward_loop(model):

    for batch in tqdm(data_loader, desc="Calibrating"):
        text_list = batch["text"]
        messages = []
        for idx in range(len(text_list)):
            messages.append(
                [{
                    "role": "user",
                    "content": text_list[idx]
                }]
            )
        text = [tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                padding=True,
            ) for msg in messages]
        model_inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda:0")

        # 前向传播
        with torch.no_grad():
            model(**model_inputs)

config = mtq.NVFP4_DEFAULT_CFG
print(config)
model = mtq.quantize(model, config, forward_loop)
print(model)

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir = export_dir,  # The directory where the exported files will be stored.
        save_modelopt_state=False,  # Whether to save the modelopt state.
    )

tokenizer.save_pretrained(export_dir)