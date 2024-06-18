import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import nms

def clip_gradients(model, max_norm=2.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0  # Accumulate the loss
    progress_bar = tqdm(data_loader)

    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # Accumulate loss

        optimizer.zero_grad()
        losses.backward()
        # Clip gradients to avoid exploding gradients
        clip_gradients(model)
        optimizer.step()

        running_loss += losses.item()  # Accumulate the loss
    
    # Calculate the average loss over the epoch
    avg_loss = running_loss / len(data_loader)

    return avg_loss

def evaluate(model, data_loader, device, label_map, iou_threshold=0.5):
    model.eval()
    coco_results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Validating'):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes = output['boxes']
                scores = output['scores']
                labels = output['labels']
                img_id = targets[i]['image_id'].item()
                
                # Apply NMS
                keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[keep].cpu().numpy()
                scores = scores[keep].cpu().numpy()
                labels = labels[keep].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if label not in label_map:
                        continue  # Skip invalid labels
                    coco_results.append({
                        "image_id": int(img_id),  # Convert to native int
                        "category_id": int(label),  # Convert to native int
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                        "score": float(score)  # Convert to native float
                    })
    return coco_results
