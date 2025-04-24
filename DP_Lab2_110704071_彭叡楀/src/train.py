import os
import argparse
import torch
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import plot_results
from evaluate import evaluate
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    #load the dataset for training & validation
    train_dataset = load_dataset(f"{args.data_path}", 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = load_dataset(f"{args.data_path}", 'valid')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    #choose the type of model you need to use then base on the device you use
    if args.model == "unet":
        model = torch.load("saved_models/unet_best.pth", map_location=device) if os.path.exists('saved_models/model_unet.pth') else UNet()
        output = "saved_models/unet.png"
    elif args.model == "resnet34":
        model = torch.load("saved_models/resnet34_best.pth", map_location=device) if os.path.exists('saved_models/model_resnet34.pth') else ResNet34_UNet()
        output = "saved_models/resnet34.png"
    else:
        raise(ValueError("Model should be 'unet' or 'resnet34'."))

    model = model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_dice = 0.0
    
    train_losses = []
    val_dices = []
    val_losses = []
    
    #train & validate the model and calculate the dice_score & accuracy (use the forward & back propagation)
    #save the model with the best dice_score
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for _, batch in enumerate(train_loader):
            images = batch['image'].to(device,dtype=torch.float)
            masks = batch['mask'].to(device,dtype=torch.float)
            
            #data augmentation: flip the photo to get different photo each time, increasing data variability
            if(args.flip == "Yes"):
                if torch.rand(1) < 0.5:
                    images = torch.flip(images, [2])
                    masks = torch.flip(masks, [2])
                
                if torch.rand(1) < 0.5:
                    images = torch.flip(images, [3])
                    masks = torch.flip(masks, [3])

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}], Train Loss: {avg_loss:.4f}")

        val_dice, val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Dice Score: {val_dice:.4f}, Validation Loss: {val_loss:.4f}")
        
        train_losses.append(avg_loss)
        val_dices.append(val_dice)
        val_losses.append(val_loss)

        if val_dice > best_dice:
            best_dice = val_dice
            model_path = os.path.join("saved_models", f"{args.model}_best.pth")
            torch.save(model, model_path)
            print(f"New best model saved to {model_path} with Dice Score: {val_dice:.4f}")

    print("Training Complete.")
    val_losses = np.array([loss.cpu().numpy() if loss.is_cuda else loss.numpy() for loss in val_losses])
    plot_results(train_losses, val_dices, val_losses, output)


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--data_path', type=str, required=True, help='Path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model type: "unet" or "resnet34"')
    parser.add_argument('--flip', '-f', type=str, default = "Yes", help='Flip the photo: "Yes" or "No"')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)
