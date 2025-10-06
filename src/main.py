import os

from model import MMLNet
from train import train
from data_set import MyDataset
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import wandb
import pickle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, help='device number')
    parser.add_argument('--model', default='MMLNet', type=str, help='the model name', choices=['MMLNet'])
    parser.add_argument('--text_name', default='', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=32, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=3, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    
    parser.add_argument('--text_mask_rate', default=0, type=float, help='text_mask_rate')
    parser.add_argument('--image_mask_rate', default=0, type=float, help='image_mask_rate')
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate for modules expect CLIP')
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, help='learning rate for CLIP')
    parser.add_argument('--max_len', default=77, type=int, help='max len of text based on CLIP')
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='../output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = set_args()
    print(torch.cuda.is_available()) 
    print(torch.version.cuda) 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    seed_everything(args.seed)

    wandb.init(
        project="MFND",
        notes="mm",
        tags=["mm"],
        config=vars(args),
    )
    wandb.watch_called = False  

    train_data = MyDataset(mode='train', text_name=args.text_name, text_mask_rate=args.text_mask_rate, image_mask_rate=args.image_mask_rate)
    dev_data = MyDataset(mode='val', text_name=args.text_name, text_mask_rate=args.text_mask_rate, image_mask_rate=args.image_mask_rate)
    test_data = MyDataset(mode='test', text_name=args.text_name, text_mask_rate=args.text_mask_rate, image_mask_rate=args.image_mask_rate)
    
    print("Number of training examples:", len(train_data))
    print("Number of validation examples:", len(dev_data))
    print("Number of test examples:", len(test_data))
    
    if args.model == 'MMLNet':
        processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/clip-vit-large-patch14")
        # processor = ChineseCLIPProcessor.from_pretrained("/root/autodl-tmp/chinese-clip-vit-large-patch14")

        model = MMLNet(args)
    else:
        raise RuntimeError('Error model name!')

    model.to(device)
    wandb.watch(model, log="all")

    train(args, model, device, train_data, dev_data, test_data, processor)



if __name__ == '__main__':
    main()
