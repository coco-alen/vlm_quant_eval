from openai import OpenAI
import time
import concurrent.futures
from tqdm import tqdm

client = OpenAI(
    base_url="http://0.0.0.0:23333/v1"
)

def call_api():
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response

# response = call_api()
# print(response.choices[0].message.content)



Total_calls = 500
max_workers = 128

# Start the timer
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future = {executor.submit(call_api): i for i in range(Total_calls)}
    for _ in tqdm(concurrent.futures.as_completed(future), total=Total_calls):
        pass
        # try:
        #     response = future[_].result()
        #     print(response.choices[0].message.content)
        # except Exception as e:
        #     print(f"Error: {e}")

end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
request_throughput = Total_calls / (end_time - start_time)
print(f"Request throughput: {request_throughput:.2f} requests/second")
