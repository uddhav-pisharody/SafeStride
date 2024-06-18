import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import utils

def run_model(image, model, device):
    model.eval()
    with torch.no_grad():
        prediction = model([image.to(device)])
    return prediction

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores=None, label_map=None):
    """
    Draws bounding boxes on the image.

    Parameters:
    - image: The image on which to draw the boxes (numpy array or similar).
    - boxes: A list of bounding boxes, each box is a list or tuple of [xmin, ymin, xmax, ymax].
    - labels: A list of labels for each bounding box.
    - scores: A list of scores for each bounding box (optional).
    - label_map: A dictionary mapping label indices to label names (optional).
    """
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    for i, box in enumerate(boxes):
        if labels[i] not in label_map:
            continue  # Skip invalid labels

        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        edgecolor = 'r' if scores is None else 'g'
        linewidth = 1 if scores is None else max(1, scores[i] * 5)
        
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
        
        if label_map is not None:
            label = label_map.get(labels[i], 'Unknown')
            score = '' if scores is None else f'{scores[i]:.2f}'
            caption = f'{label} {score}'
            ax.text(xmin, ymin, caption, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.show()

def display_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
