import argparse
from oxford_pet import load_dataset
import torch
from torch.utils.data import DataLoader
from utils import dice_score
import numpy as np
from PIL import Image
import os
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_mask(mask, output_path, index):
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(f"{output_path}/mask_{index:03d}.png")

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
    num_batches = 0
    
    #get the predicted answer,dice score and loss
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device, dtype=torch.float)
            masks = batch['mask'].to(device, dtype=torch.float)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.float()

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            dice = dice_score(preds, masks)
            total_dice += dice

            for j in range(preds.size(0)):
                mask = preds[j].cpu().numpy().squeeze()
                save_mask(mask, output_path, i * args.batch_size + j)

            num_batches += 1

    avg_dice = total_dice / num_batches
    avg_loss = total_loss / num_batches

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    result_path = os.path.join(output_path, "results.txt")
    with open(result_path, "w") as f:
        f.write(f"Average Dice Score: {avg_dice:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")

    print(f"Results saved to {result_path}")


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='unet', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    test(args)
