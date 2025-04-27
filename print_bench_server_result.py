import os
import json
import pandas as pd


result_dir = "result/Qwen2.5-VL-3B-Instruct-w8a8fp8-benchBatch2"
result = pd.DataFrame(columns=["Input-len", "output-len", "max-concurrency", "prefill-throughput", "output-throughput", "TTFT(ms)", "TPOT(ms)"])
def get_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


all_result_json = get_all_files(result_dir)
for result_json in all_result_json:
    output_len = int(result_json.split("/")[-1].split("_")[-1][:-6])
    input_len = int(result_json.split("/")[-1].split("_")[-2])
    with open(result_json, 'r', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            max_concurrency = data["max_concurrency"]
            prefill_throughput = data["input_throughput"]
            output_throughput = data["output_throughput"]
            TTFT = data["mean_ttft_ms"]
            TPOT = data["mean_tpot_ms"]

            new_row = pd.Series([input_len, output_len, max_concurrency, prefill_throughput, output_throughput, TTFT, TPOT], index=result.columns)
            result = result._append(new_row, ignore_index=True)
            
result = result.sort_values(by=["Input-len", "output-len", "max-concurrency"], ascending=[True, True, True])
print(result)

