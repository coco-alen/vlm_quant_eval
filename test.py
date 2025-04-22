from openai import OpenAI
client = OpenAI(
    api_key="miniphant",
    base_url="http://0.0.0.0:23333/v1"
)
print(client.models.list().data[0].id)