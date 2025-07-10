import tkinter as tk
import time
from tkinter import messagebox, scrolledtext
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======== Load Model (PyTorch version) ========
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = "C:/Users/KHADEER KHAN/OneDrive/Documents/lama"

print("🔄 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
model.eval()
print("✅ Model loaded.")

# ======== GUI Setup ========
root = tk.Tk()
root.title("🧠 TinyLLaMA Chat (PyTorch CPU)")
root.geometry("720x520")

frame = tk.Frame(root)
frame.pack(pady=10)

entry = tk.Entry(frame, width=70, font=("Arial", 13))
entry.pack(side=tk.LEFT, padx=10)

button = tk.Button(frame, text="Ask", font=("Arial", 12), bg="#a6e1fa")
button.pack(side=tk.LEFT)

output_text = scrolledtext.ScrolledText(root, height=20, width=85, font=("Courier", 11))
output_text.pack(padx=10, pady=10)
output_text.config(state="disabled")

def ask_question():
    question = entry.get().strip()
    if not question:
        messagebox.showwarning("Warning", "Please enter a question.")
        return

    prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    output_text.config(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"👤 You: {question}\n\n🤖 Answer: ")
    output_text.config(state="disabled")
    root.update()

    # Start timing
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # End timing
    end_time = time.time()
    elapsed = round(end_time - start_time, 2)

    full_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if question in full_reply:
        answer = full_reply.split(question, 1)[-1].strip()
    else:
        answer = full_reply.strip()

    tokens = len(tokenizer.encode(answer))
    speed = round(tokens / elapsed, 2) if elapsed > 0 else 0

    # Display result
    output_text.config(state="normal")
    output_text.insert(tk.END, answer)
    output_text.insert(tk.END, f"\n\n⏱️ Time: {elapsed}s\n⚡ Speed: {speed} tokens/s\n🔢 Tokens: {tokens}")
    output_text.config(state="disabled")

button.config(command=ask_question)
root.mainloop()
