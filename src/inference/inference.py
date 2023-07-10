#!/usr/bin/env python3
# real-time water bottle inspection from a camera feed with an exported ONNX model
import argparse
import os
import cv2
import onnxruntime
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# parse command line
parser = argparse.ArgumentParser(description='Classify water bottles from a camera feed in real time')
parser.add_argument('model', help='path to the ONNX model')
parser.add_argument('--camera', default=0, type=int, help='camera index for cv2.VideoCapture (default: 0)')
parser.add_argument('--resolution', default=299, type=int, help='input NxN resolution of the model (default: 299 for Inception)')
parser.add_argument('--classes', default='Defective,Propre', help='comma-separated class names in model output order (default: Defective,Propre)')
args = parser.parse_args()

# Use TensorRT / CUDA when available (Jetson Nano), fall back to the CPU otherwise
providers = [p for p in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
             if p in onnxruntime.get_available_providers()]
session = onnxruntime.InferenceSession(args.model, providers=providers)
print(f"=> running on {session.get_providers()}")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define the image preprocessing transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(args.resolution),
    transforms.CenterCrop(args.resolution),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the list of class names
class_names = args.classes.split(',')

# Initialize the camera
cap = cv2.VideoCapture(args.camera)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("=> could not read from the camera")
        break

    # Convert the frame to RGB format and to a tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess(frame_rgb).unsqueeze(0)

    # Perform inference
    output = session.run([], {input_name: frame_tensor.numpy()})[0]

    # Get the predicted class and confidence
    pred_idx = output.argmax()
    pred_class = class_names[pred_idx]
    confidence = max(output[0])

    # Print the results to console
    print(f"class={pred_class}, confidence={confidence:.2f}")

    # Display the frame with the predicted class and confidence printed on it
    cv2.putText(frame, f"{pred_class} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Water bottle inspection', frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
