import torch
import matplotlib.pyplot as plt
import os

def dice_score(pred_mask, gt_mask, eps=1e-7):

    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()
    
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    dice = (2.0 * intersection + eps) / (union + eps)
    
    return dice.item()

def plot_results(train_losses, val_dices, val_losses, output):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    #training loss curve
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #validation loss curve
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='red')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    #dice score curve
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_dices, label='Validation Dice Score', marker='o', color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    print(f"Plot saved to {output}")
    
    plt.close()