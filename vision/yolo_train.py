from torch.utils.data import DataLoader
from vision.coco_dataset import COCODataset
from ultralytics import YOLO
import os
import argparse
import torch


class YOLOTrainer():
    def __init__(self, dataset: COCODataset, batch_size: int = 8, model = None, args: argparse.Namespace = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=dataset.collate_fn())
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def obj_detect_train(self):
        # Iterating through the dataloader
        model = YOLO(os.path.join(os.getcwd(), "models", "YOLO", "yolo11n.pt"))
        model.to(self.device)
        print(self.device)
        train_results = model.train(
            data=os.path.join(self.args.root, "coco.yaml"),
            epochs=100,
            imgsz=640,
            device=self.device,
            workers=16,
            batch=16,
        )


        metrics = model.val()


        results = model("path/to/image.jpg")
        results[0].show()


        path = model.export(format="onnx") 