import gradio as gr
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import cv2

# Load your trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('models/faster_rcnn_resnet50.pth', map_location=device)
model.eval()

def get_prediction(image):
    # Preprocess the image
    image = Image.fromarray(image)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract the bounding boxes, labels, and scores
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    return boxes, labels, scores

def draw_boxes(image, boxes, labels, scores, label_map=None):
    image = np.array(image)

    for i, box in enumerate(boxes):
        if label_map and labels[i] not in label_map:
            continue  # Skip invalid labels

        xmin, ymin, xmax, ymax = box
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        
        label = label_map[labels[i]] if label_map else 'Person'
        score = f'{scores[i]:.2f}' if scores is not None else ''
        caption = f'{label} {score}'
        
        cv2.putText(image, caption, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def predict(image):
    boxes, labels, scores = get_prediction(image)
    label_map = {0: 'Pedestrians', 1: 'child', 2: 'person', 3: 'silver', 4: 'wheel'}  # Adjust label map based on your dataset
    result_image = draw_boxes(image, boxes, labels, scores, label_map)
    return result_image

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="numpy", label="Upload Image"),
    outputs=gr.outputs.Image(type="numpy", label="Detected Image"),
    title="SafeStride Pedestrian Detection",
    description="Upload an image to detect pedestrians using the Faster R-CNN model."
)

if __name__ == "__main__":
    iface.launch()
