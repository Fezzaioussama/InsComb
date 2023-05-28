import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import torch
from torchvision import models, transforms
import onnx
import onnxruntime
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# Load the ONNX model
model = onnx.load(r"C:\Users\taibeche ahmed\Desktop\project\pytorch-classification-master\models\incept_lr_0.001_same_wddrop_d2\inceptdata1_ep80")

# Create a PyTorch model from the ONNX model
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#providers = ['CUDAExecutionProvider']
#pytorch_model = onnxruntime.InferenceSession(r"C:\Users\taibeche ahmed\Desktop\project\pytorch-classification-master\models\res50\res50largedata", options, providers)
# Set available providers
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']

# Create InferenceSession object with specified providers
# Create InferenceSession object with specified providers

pytorch_model = onnxruntime.InferenceSession(r"C:\Users\taibeche ahmed\Desktop\project\pytorch-classification-master\models\incept_lr_0.001_same_wddrop_d2\inceptdata1_ep80",
                                             providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])


input_name = pytorch_model.get_inputs()[0].name
output_name = pytorch_model.get_outputs()[0].name

# Define the image preprocessing transforms
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the list of class names
class_names = ['Defective', 'Propre']

# Choose the input source
#input_source = input("Choose the input source ('webcam' or 'folder'): ")
input_source='webcam'
if input_source == 'webcam':
    # Initialize the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to tensor

        frame_tensor = preprocess(frame_rgb).unsqueeze(0)
        # Move tensor to GPU device

        frame_tensor = frame_tensor.to('cuda')
        # Perform inference

        with torch.no_grad():
            output = pytorch_model.run([], {input_name: frame_tensor.cpu().numpy()})[0]
        '''
        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       # Convert the frame to tensor and preprocess
        with torch.no_grad():
            frame_tensor = preprocess(frame_rgb).unsqueeze(0)

            # Perform inference
            output = pytorch_model.run([], {input_name: frame_tensor.numpy()})[0]
        '''









        # Get the predicted class and accuracy
        pred_idx = output.argmax()
        pred_class = class_names[pred_idx]
        accuracy = max(output[0])

        # Print the results to console
        print(f"class={pred_class}, accuracy={accuracy:.2f}")

        # Display the frame with predicted class and accuracy printed on it
        plt.imshow(frame_rgb)
        plt.text(10, 10, f"{pred_class} ({accuracy:.2f})", color='white')
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()

        # Check for quit command
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

