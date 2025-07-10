from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID and custom cache directory
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = "C:/Users/KHADEER KHAN/OneDrive/Documents/lama"

# Download and cache the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

print("✅ TinyLlama model and tokenizer downloaded successfully.")
