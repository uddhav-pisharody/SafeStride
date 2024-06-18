import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import os
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

        # Create a mapping from category IDs to a consecutive range of labels (starting from 1)
        self.cat_id_to_label = {cat['id']: idx + 1 for idx, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img = coco.loadImgs(img_id)[0]
        path = img['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        if len(anns) == 0:
            # Skip images with no annotations
            return None

        boxes = []
        labels = []
        for ann in anns:
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def load_data(root, annotation_file, batch_size, collate_fn):
    train_dataset = CocoDataset(
        root,
        annotation_file,
        transform=get_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return train_loader
