import torch
import engine
from utils import collate_fn
from loader import load_data
from model import get_model
from loader import CocoDataset, get_transform

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data
    root = 'data'
    annotation_file = 'anns'
    batch_size = 8
    dataset = CocoDataset(root,annotation_file,transform=get_transform())
    data_loader = load_data(root, annotation_file, batch_size, collate_fn=collate_fn)
    label_map = {cat['id']: cat['name'] for cat in dataset.coco.loadCats(dataset.coco.getCatIds())}

    # Initialize model
    num_classes = 6  # Background and pedestrians(5 kinds)
    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train model
    num_epochs = 10
    for epoch in range(num_epochs):
        engine.train_one_epoch(model, optimizer, data_loader, device, epoch)
        # Save model after every epoch
        torch.save(model.state_dict(), f'models/model_epoch_{epoch}.pth')

if __name__ == "__main__":
    main()
