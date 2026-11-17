import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the model you just trained
model = joblib.load('asl_letter_model.pkl')

# Tell the converter what shape of input to expect: 
# None = "any number of rows", 63 = your 21 landmarks x 3 coordinates
initial_type = [('float_input', FloatTensorType([None, 63]))]

# Convert
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save it
with open('asl_letter_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Saved asl_letter_model.onnx")