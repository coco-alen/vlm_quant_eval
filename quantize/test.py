from transformers import AutoProcessor, AutoModelForImageTextToText
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# print(processor)

dataset = load_dataset("lmms-lab/MME" , split="test")
# 设置 batch_size 和最大样本数
batch_size = 1
calib_size = 2 * batch_size

# 取一部分数据用于校准
dataset = dataset.select(range(calib_size))
def collate_fn(batch):
    texts = [item['question'] for item in batch]
    images = [item['image'].convert('RGB') for item in batch]  # 确保图像是 RGB 格式
    return {"text": texts, "image": images}

# 创建 DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
)

for batch in tqdm(data_loader, desc="Calibrating"):
    text_list = batch["text"]
    image_list = batch["image"]
    messages = []
    for idx in range(len(text_list)):
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_list[idx],
                    },
                    {"type": "text", "text": text_list[idx]},
                ],
            }
        )
    text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print("===============================")
    print(inputs["input_ids"].shape)
    print(inputs["pixel_values"].shape)