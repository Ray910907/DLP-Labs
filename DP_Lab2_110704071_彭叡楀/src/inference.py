import argparse
from oxford_pet import load_dataset
import torch
from torch.utils.data import DataLoader
from utils import dice_score
import numpy as np
from PIL import Image
import os
import torch.nn as nn
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    #load the dataset for testing
    dataset = load_dataset(f"{args.data_path}", "test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    #choose the model you want to use and load the model we already save, turn it into validate mode
    if args.model == "unet":
        model = torch.load("saved_models/unet_best.pth", map_location=device)
        output_path = f"saved_models/unet_predictions"
    elif args.model == "resnet34":
        model = torch.load("saved_models/resnet34_best.pth", map_location=device)
        output_path = f"saved_models/resnet34_predictions"
    else:
        raise(ValueError("Model should be 'unet' or 'resnet34'."))
    
    model.device = device
    model.eval()

    
    os.makedirs(output_path, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    total_dice = 0
    total_loss = 0
    
    #get the predicted answer,dice score and loss
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            images = batch['image'].to(device, dtype=torch.float)
            masks = batch['mask'].to(device, dtype=torch.float)

            outputs = model(images)
            preds = (torch.round(torch.sigmoid(outputs))).float()

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            dice = dice_score(preds, masks)
            total_dice += dice

            for j in range(preds.size(0)):
                mask = preds[j].cpu().numpy().squeeze()
                id = i * args.batch_size + j
                mask = mask = np.round(mask * 255).astype(np.uint8)
                mask = Image.fromarray(mask)
                mask.save(f"{output_path}/{id}_mask.png")

    avg_dice = total_dice / len(dataloader)
    avg_loss = total_loss / len(dataloader)

    print(f"Dice Score: {avg_dice:.4f} Average Loss: {avg_loss:.4f}\n")


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='unet', help='Model type: "unet" or "resnet34"')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    test(args)
