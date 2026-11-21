import joblib
import numpy as np
import onnxruntime as ort   # pip install onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load('asl_letter_model.pkl')

# 77 = 63 landmark values + 14 distance features
initial_type = [('float_input', FloatTensorType([None, 77]))]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={id(model): {'zipmap': False}},
)

with open('asl_letter_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
print("Saved asl_letter_model.onnx")
print("Class order:", list(model.classes_))

# Quick check: ONNX and the original model should agree
sess = ort.InferenceSession('asl_letter_model.onnx')
X = np.random.randn(200, 77).astype(np.float32)
labels, probs = sess.run(None, {'float_input': X})
print("Same labels:", (labels == model.predict(X)).mean())
print("Max probability difference:", np.abs(probs - model.predict_proba(X)).max())