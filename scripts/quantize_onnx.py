from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("models/onnx")

qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

quantizer.quantize(save_dir="models/onnx_quantized", quantization_config=qconfig)

print("Quantized model saved to models/onnx_quantized")
