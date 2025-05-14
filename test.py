import base64
import os
from openai import OpenAI
import time
import concurrent.futures
from tqdm import tqdm

client = OpenAI(
    base_url="http://127.0.0.1:23333/v1"
)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def call_api():

    base64_image = encode_image("/mnt/raid0/yipin/quant_eval/example_image.png")
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #     "text": "What is in this image?",
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{base64_image}",
                    #     },
                    # },
                    {
                        "type": "text",
                        "text": """
Resolution theorem proving for showing that a formula of propositional logic is not satisfiable has which of the following properties?
I. It is a sound proof system in the sense that there does not exist a proof of the unsatisfiability of a satisfiable formula of propositional logic.
II. It is a complete proof system in the sense that there is a proof of unsatisfiability for every unsa tisfiable formula of propositional logic.
III. It is a succinct proof system in the sense that whenever an unsatisfiable formula F of propositional logic has a resolution proof, F also has a proof whose length is polynomial in the length of F.
Here you can choose from: [ "I only", "III only", "I and II only", "I and III only" ].
""",
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    print(response.choices[0].message.content)
    return response

# response = call_api()



Total_calls = 5120
max_workers = 5120

# Start the timer
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_item = {executor.submit(call_api): i for i in range(Total_calls)}
    for future in tqdm(concurrent.futures.as_completed(future_to_item), total=Total_calls):
        idx = future_to_item[future]
        try:
            future.result()  
        except Exception as exc:
            print(f'Error processing sample {idx}: {exc}')


end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
request_throughput = Total_calls / (end_time - start_time)
print(f"Request throughput: {request_throughput:.2f} requests/second")