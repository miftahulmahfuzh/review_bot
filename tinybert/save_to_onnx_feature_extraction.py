from pathlib import Path

# from optimum.onnxruntime import ORTModelForSequenceClassification
# from optimum.onnxruntime import ORTModelForMaskedLM
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

base_model = "."
onnx_path = Path(".")

# load vanilla transformers and convert to onnx
# model = ORTModelForSequenceClassification.from_pretrained(base_model, from_transformers=True)
# model = ORTModelForMaskedLM.from_pretrained(base_model, from_transformers=True)
model = ORTModelForFeatureExtraction.from_pretrained(base_model, from_transformers=True)
# tokenizer = AutoTokenizer.from_pretrained(base_model)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
# tokenizer.save_pretrained(onnx_path)
