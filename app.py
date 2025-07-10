import numpy as np
from openvino import Core
from transformers import AutoTokenizer
import gradio as gr
import speech_recognition as sr
import threading

# Load OpenVINO model
core = Core()
model_dir = r"C:\Users\KHADEER KHAN\OneDrive\Documents\lama\tinyllama_ir_fp16"
compiled_model = core.compile_model(f"{model_dir}/tinyllama_fp16.xml", "CPU")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\Users\KHADEER KHAN\OneDrive\Documents\lama\tinyllama_onnx_past"
)

def detect_num_layers(compiled_model):
    i = 0
    while True:
        try:
            compiled_model.input(f"past_key_values.{i}.key")
            i += 1
        except RuntimeError:
            break
    return i

num_layers = detect_num_layers(compiled_model)

def init_empty_past():
    past = {}
    for i in range(num_layers):
        k = compiled_model.input(f"past_key_values.{i}.key")
        v = compiled_model.input(f"past_key_values.{i}.value")
        k_shape = [1 if d.is_dynamic else int(d.get_length()) for d in k.partial_shape]
        v_shape = [1 if d.is_dynamic else int(d.get_length()) for d in v.partial_shape]
        k_shape[2] = 0
        v_shape[2] = 0
        past[f"past_key_values.{i}.key"] = np.zeros(k_shape, dtype=np.float32)
        past[f"past_key_values.{i}.value"] = np.zeros(v_shape, dtype=np.float32)
    return past

def clean_decode(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True).replace("*", "").replace("~", "")

def fix_prompt_punctuation(text):
    if text and text[-1] not in [".", "?", "!"]:
        return text + "."
    return text

# Stop flag
stop_flag = threading.Event()

def generate_reply_stream(user_prompt):
    prompt = f"### Human:\n{user_prompt.strip()}\n\n### Assistant:\n"
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    generated_ids = input_ids.copy()
    past_key_values = init_empty_past()
    prev_text = clean_decode(generated_ids[0])

    inputs = {
        "input_ids": input_ids,
        "attention_mask": np.ones_like(input_ids),
        "position_ids": np.arange(input_ids.shape[1]).reshape(1, -1)
    }
    inputs.update(past_key_values)

    outputs = compiled_model(inputs)
    logits = outputs[compiled_model.output("logits")]

    for i in range(num_layers):
        past_key_values[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
        past_key_values[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

    next_token_id = int(np.argmax(logits[0, -1, :]))
    generated_ids = np.concatenate([generated_ids, [[next_token_id]]], axis=1)
    prev_text = clean_decode(generated_ids[0])
    yield prev_text

    for _ in range(500):
        if stop_flag.is_set():
            break
        last_token = generated_ids[:, -1:]
        position_ids = np.array([[generated_ids.shape[1] - 1]])
        attention_mask = np.ones_like(generated_ids)

        inputs = {
            "input_ids": last_token,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        inputs.update(past_key_values)

        outputs = compiled_model(inputs)
        logits = outputs[compiled_model.output("logits")]
        next_token_id = int(np.argmax(logits[0, -1, :]))

        if next_token_id == tokenizer.eos_token_id:
            break

        generated_ids = np.concatenate([generated_ids, [[next_token_id]]], axis=1)

        for i in range(num_layers):
            past_key_values[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
            past_key_values[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

        current_text = clean_decode(generated_ids[0])
        new_text = current_text[len(prev_text):]
        prev_text = current_text
        yield new_text

        if "<|end|>" in new_text or "</s>" in new_text:
            break

def transcribe_audio(audio_file):
    if not audio_file or not isinstance(audio_file, str):
        return "No audio file selected."
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data)
    except Exception as e:
        return f"Error: {str(e)}"

custom_css = """
body, .gradio-container {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif;
}
.chatbot .message.user {
    background-color: #1f6feb;
    color: white;
    border-radius: 10px;
}
.chatbot .message.assistant {
    background-color: #161b22;
    color: #c9d1d9;
    border-radius: 10px;
}
textarea, input, .btn {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}
.gr-button {
    background-color: #238636 !important;
    color: white !important;
    border-radius: 8px;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.HTML("<link href='https://fonts.googleapis.com/css2?family=Poppins&display=swap' rel='stylesheet'>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("logo.png", label="", show_label=False, width=80)  # Replace with your logo
        with gr.Column(scale=5):
            gr.Markdown("## 🤖Study Buddy🎓")
            gr.Markdown("Ask me anything! I'm your compact AI assistant running fast on Intel CPUs.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Chat", height=480, elem_classes="chatbot", type="messages")
            user_input = gr.Textbox(placeholder="Type something...", label="Your message", lines=2)
            with gr.Row():
                send_btn = gr.Button("💡 Send")
                stop_btn = gr.Button("🛑 Stop")
                clear_btn = gr.Button("🧹 Clear")

        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎙 Upload or Record Audio")
            transcribe_btn = gr.Button("🎧 Transcribe Audio")

    def handle_chat(history, message):
        stop_flag.clear()
        if not message.strip():
            return history, ""
        history = history or []
        history.append({"role": "user", "content": message})
        partial = ""
        for chunk in generate_reply_stream(message):
            partial += chunk
            yield history + [{"role": "assistant", "content": partial}], ""

    def stop_response():
        stop_flag.set()
        return gr.update()

    send_btn.click(fn=handle_chat, inputs=[chatbot, user_input], outputs=[chatbot, user_input])
    stop_btn.click(fn=stop_response, outputs=[chatbot])
    clear_btn.click(lambda: [], None, chatbot)
    transcribe_btn.click(fn=transcribe_audio, inputs=audio_input, outputs=user_input)

demo.launch()
