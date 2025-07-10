
# 🤖 Study Buddy

A lightweight, CPU-optimized AI assistant built using OpenVINO and TinyLLaMA. It features a Gradio-powered UI, real-time streaming responses, and voice input via speech recognition. Perfect for offline use, study assistance, or privacy-first AI deployment.

Demo UI:
🧠 Study Buddy — ask anything, text or voice!

GitHub by: [Nadeem Khan](https://github.com/Nadeem101104)

---

## 📦 Features

* ✅ Local language model inference with OpenVINO (no internet required after setup)
* ✅ Real-time token-by-token streaming responses
* ✅ Chat interface with clear, modern UI (Gradio)
* ✅ Voice input support (record or upload audio)
* ✅ Runs on standard CPUs (ideal for Intel hardware)
* ✅ Stop/Interrupt button for responsive control
* ✅ Custom dark theme for better visual experience

---

## 🧠 Tech Stack

| Component           | Technology               |
| ------------------- | ------------------------ |
| Model Inference     | OpenVINO Runtime         |
| Language Model      | TinyLLaMA 1.1B Chat      |
| Tokenization        | HuggingFace Transformers |
| UI Framework        | Gradio                   |
| Audio Transcription | SpeechRecognition        |
| Deployment          | Python (local script)    |

---

## 🚀 How to Run

1. Clone the repo:

   git clone [https://github.com/Nadeem101104/StudyBuddy-AI](https://github.com/Nadeem101104/Study-Buddy)

2. Install dependencies:

pip install transformers==4.40.1

pip install openvino==2023.3.0

pip install "optimum[exporters]==1.17.1"

pip install numpy>=1.23.5

pip install gradio==4.26.0

pip install SpeechRecognition==3.10.1

pip install torch==2.2.2

pip install torchaudio==2.2.2

pip install scipy>=1.11.3

pip install pyaudio==0.2.13

pip install setuptools>=65.5.1

pip install huggingface-hub>=0.21.3

3. Download & export TinyLLaMA model (ONNX to IR format):

   a. Export to ONNX:

   python -m optimum.exporters.onnx&#x20;
   \--model TinyLlama/TinyLlama-1.1B-Chat-v1.0&#x20;
   \--task text-generation-with-past&#x20;
   \--device cpu&#x20;
   \--cache\_dir "your/model/cache/path"&#x20;
   "your/output/onnx/folder"

   b. Convert ONNX to IR (FP16):

   python -m openvino.tools.ovc&#x20;
   "your/output/onnx/folder/model.onnx"&#x20;
   \--compress\_to\_fp16&#x20;
   \--output\_model "your/output/ir/folder/tinyllama\_fp16.xml"

4. Run the chatbot:

   python app.py

---

## 📁 Folder Structure

| Folder/File              | Description                  |
| ------------------------ | ---------------------------- |
| app.py                   | Main chatbot script          |
| tinyllama\_fp16.xml/.bin | OpenVINO IR model (FP16)     |
| tinyllama\_onnx\_past    | Tokenizer and ONNX artifacts |
| logo.png                 | Logo used in the interface   |
| requirements.txt         | Python dependencies          |
| README.md                | This documentation           |

---

## 🎯 Use Cases

* Study assistant for school and college students
* Offline AI assistant for developers
* Chatbot for constrained hardware (Intel NUCs, laptops)
* Voice-to-text + AI helper combo for accessibility
* Low-cost, private local AI experiments

---

## 📌 Customization Ideas

* Add TTS (text-to-speech) for voice replies
* Add authentication and user profiles
* Store conversation history per session
* Add image-based input with vision models
* Deploy as desktop app using Electron

---

## 🛠️ Troubleshooting

* If OpenVINO model fails to load, recheck IR conversion paths and file names.
* For SpeechRecognition to work, ensure you’re connected to the internet (Google API used).
* If Gradio UI doesn't load, try a different browser or update Gradio.
* Run Python as administrator if symlink warnings appear during model export.

---

## ✨ Credits

* Model: TinyLLaMA-1.1B Chat (by TinyLlama project)
* Optimization: OpenVINO Toolkit by Intel
* UI: Gradio team
* Tokenization: Hugging Face
* Voice Input: SpeechRecognition (Google API)

---

## 📄 License

This project is open-source under the MIT License. See LICENSE file for details.

---

📌 GitHub Repository:
[https://github.com/Nadeem101104/StudyBuddy-AI](https://github.com/Nadeem101104/Study-Buddy)

