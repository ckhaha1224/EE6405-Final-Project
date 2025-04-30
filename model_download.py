from transformers import AutoTokenizer, AutoModelForCausalLM

# Model Path
MODEL_PATH = "vvEverett/EmoLLM_Qwen"
LOCAL_PATH = "./model_Emo-Qwen/"

# Download and load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load Model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')

# Save the model locally
print("Save Model to", LOCAL_PATH)
tokenizer.save_pretrained(LOCAL_PATH)
model.save_pretrained(LOCAL_PATH)
print("Done!")