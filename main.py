from vision.coco_dataset import COCODataset
from vision.yolo_train import YOLOTrainer
import argparse
from utils import globals
from utils import utils
import config

import os
os.system("color")



#########################################################################
#
# Sets default configurations for command-line arguments to reduce the 
# need for specifying many arguments manually.
#
#########################################################################
def set_config(args):
    user_args = config.PATHS[args.usr]
    cmd_args = vars(args)
    
    for key, val in user_args.items():
        cmd_args[key] = val

    args = argparse.Namespace(**cmd_args)
    return args


#########################################################################
#
# Setup
#
#########################################################################
def setup(args):
    if args.create_splits:
        utils.create_yolo_splits(args.root, split="train2017", amt=5)
        utils.create_yolo_splits(args.root, split="test2017", amt=5)
        utils.create_yolo_splits(args.root, split="val2017", amt=5)
#########################################################################
#
# Sets default configurations for command-line arguments to reduce the 
# need for specifying many arguments manually.
#
#########################################################################
def main(args):
    setup(args)
    coco = COCODataset(args.root, split="train")
    coco_val = COCODataset(args.root, split="val")
    coco.coco_to_yolo()
    coco_val.coco_to_yolo()
    trainer = YOLOTrainer(coco, args=args)
    trainer.obj_detect_train()


#########################################################################
#
# Main entry point of the system.
#
#########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry Point of VQA Algo")
    parser.add_argument("--usr", type=str, default="win", help="set default user")
    parser.add_argument("--root", type=str, help="path/to/root/dir")
    parser.add_argument("--train", type=str, help="Start training loop")
    parser.add_argument("--create_splits", action="store_true", help="Creates train/val/test splits")
    
    args = parser.parse_args()
    args = set_config(args)
    
    utils.print_torch_info()
    main(args)