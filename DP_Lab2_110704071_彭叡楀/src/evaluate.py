import argparse
from oxford_pet import load_dataset
import torch
from torch.utils.data import DataLoader
from utils import dice_score
import numpy as np
from PIL import Image
import os

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(net, data, device,criterion):
    #turn the model into the evaluate mode
    net.eval()
    total_dice = 0
    total_loss = 0
    num_batches = 0
    #caculate the average accuracy & dice score of each data by model forwarding of image and the mask
    with torch.no_grad():
        for _ , batch in enumerate(data):
            images = batch['image'].to(device,dtype=torch.float)
            masks = batch['mask'].to(device,dtype=torch.float)

            
            outputs = net(images)
            preds = torch.sigmoid(outputs) > 0.5

            dice = dice_score(preds, masks)
            total_dice += dice

            loss = criterion(outputs, masks)
            total_loss += loss

            num_batches += 1

    avg_dice = total_dice / num_batches
    avg_loss = total_loss / num_batches
    
    
    return avg_dice, avg_loss

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet model on the dataset')
    parser.add_argument('--model', default='saved_models/model_UNet.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')

    return parser.parse_args()
