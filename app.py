# from openai import OpenAI

# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = ""
# )

# completion = client.chat.completions.create(
#   model="meta/llama3-70b-instruct",
#   messages=[{"role":"user","content":"hello"}],
#   temperature=0.5,
#   top_p=1,
#   max_tokens=1024,
#   stream=True
# )

# for chunk in completion:
#   if chunk.choices[0].delta.content is not None:
#     print(chunk.choices[0].delta.content, end="")


from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"Provide me an article on Machine Learning."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")