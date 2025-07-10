python -m optimum.exporters.onnx ^
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --task text-generation-with-past ^
  --device cpu ^
  --cache_dir "C:\Users\KHADEER KHAN\OneDrive\Documents\lama" ^
  "C:\Users\KHADEER KHAN\OneDrive\Documents\lama\tinyllama_onnx_past"