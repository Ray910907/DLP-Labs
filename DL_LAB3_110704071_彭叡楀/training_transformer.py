import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.train_loss_min = float('inf')
        self.val_loss_min = float('inf')
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self,epoch,device,accum_grad,train_loader):
        self.model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
        for i, data in enumerate(loop):
            #get latent and probability token
            logits, z_indices = self.model(data.to(device))
            #compute the cross-entropy loss between the logits and target z_indices, then use it to do back propagation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            loss /= accum_grad
            loss.backward()

            #update the model parameters after accumulating gradients for a specific number of steps
            if (i + 1) % accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            
            total_loss += loss.item()

            loop.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{(total_loss / (i + 1)):.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        self.scheduler.step() 
        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")

    def eval_one_epoch(self,epoch,device,val_loader):
        self.model.eval()
        total_loss = 0
        loop = tqdm(val_loader, desc=f"Evaluating Epoch {epoch}", leave=True)

        with torch.no_grad():
            for i, data in enumerate(loop):
                logits, z_indices = self.model(data.to(device))
                
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
                total_loss += loss.item()

                loop.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{(total_loss / (i + 1)):.4f}"
                })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        #save the model if the current training loss is smaller than the minimum one
        if(self.val_loss_min > avg_loss):
            self.val_loss_min = avg_loss
            torch.save(self.model.transformer.state_dict(), "./transformer_checkpoints/best_val.pth")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_transformer.train_one_epoch(epoch,args.device,args.accum_grad,train_loader)
        train_transformer.eval_one_epoch(epoch,args.device,val_loader)