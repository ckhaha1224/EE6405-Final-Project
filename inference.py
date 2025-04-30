import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model_Emollama-chat-7b/")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./model_Emollama-chat-7b/", device_map='auto')
model.eval()

# 示例提示
prompt = '''
Human:
Task: Categorize the text into an ordinal class that best characterizes the writer's mental state, considering various degrees of positive and negative sentiment intensity. 3: very positive mental state can be inferred. 2: moderately positive mental state can be inferred. 1: slightly positive mental state can be inferred. 0: neutral or mixed mental state can be inferred. -1: slightly negative mental state can be inferred. -2: moderately negative mental state can be inferred. -3: very negative mental state can be inferred
Text: Beyoncé resentment gets me in my feelings every time. 😩
Intensity Class:

Assistant:

'''

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs["input_ids"], max_length=256)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)