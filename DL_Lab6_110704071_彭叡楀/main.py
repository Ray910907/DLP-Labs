import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from diff_dataloader import Dataset
from DDPM import DDPM
from evaluator import evaluation_model
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
import numpy as np
import argparse
import os

@torch.no_grad()
def test(ddpm_model, noise_scheduler, dataloader, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm_model.net.eval()
    accs = []
    result = []
    evaluator = evaluation_model()
    for _, label in enumerate(tqdm(dataloader)):
        progress = []
        label = label.to(device)
        if not os.path.exists(out):
            os.makedirs(out)

        img = torch.randn(1, 3, 64, 64).to(device)
        for i, t in enumerate(noise_scheduler.timesteps):
            predicted_noise = ddpm_model.forward(img, t, label)

            img = noise_scheduler.step(predicted_noise, t, img).prev_sample

            if i % 100 == 0:
                progress.append(img.squeeze(0))
        
        progress.append(img.squeeze(0))
        result.append(img.squeeze(0))
        progress = torch.stack(progress)
        grid = make_grid((progress + 1) / 2,nrow=len(progress))
        save_image(grid, f'./{out}/processing_{_}.jpg')
        save_image((img + 1) / 2, f'./{out}/{_}.png')
        
        
        acc = evaluator.eval(img, label)
        accs.append(acc)
        #print(label)
        print(f"Accuracy: {acc:.4f}")
    result = torch.stack(result)
    test_grid = make_grid((result + 1) / 2, nrow=8)
    save_image(test_grid, f'{out}/test_result.jpg')
    print(f"Average Accuracy: {np.mean(accs):.4f}")


def train(ddpm_model, dataloader, optimizer, noise_scheduler, scheduler, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm_model.net.train()
    losses = []
    best_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        for _, (img, label) in enumerate(tqdm(dataloader)):
            img = img.to(device)
            label = label.to(device)

            noise = torch.randn_like(img).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (img.size(0),), device=device).long()
            noisy_img = noise_scheduler.add_noise(img, noise, timesteps)

            pred = ddpm_model.forward(noisy_img, timesteps, label)

            loss = F.mse_loss(pred, noise)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(ddpm_model.net.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.5f}")
        losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': ddpm_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'best_train.pth')
    
    plt.plot(losses, marker='o', color='blue', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig('Training_loss_2500.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root',     type=str, default="iclevr",  help="Your Dataset Path")
    parser.add_argument('--checkpoint',     type=str, default="best_train.pth",  help="Your Checkpoint Name")
    parser.add_argument('--out_root',     type=str, default="output",  help="Your Output Path")
    parser.add_argument('--lr',     type=float, default=1e-4,  help="Learning Rate")
    parser.add_argument('--mode',     type=str, default='train',  help="Mode of Train or Test (Input train, test, new_test)")
    parser.add_argument('--num_epoch',     type=int, default=40,  help="Epoch to training")
    parser.add_argument('--beta_end',     type=float, default=0.02,  help="Epoch to training")
    parser.add_argument('--num_train_timesteps',     type=int, default=1000,  help="Epoch to training")

    args = parser.parse_args()
    batch_size = 2
    num_epochs = 40
    lr = 1e-4
    num_classes = 24
    
    dataset = Dataset(image_dir=args.in_root,mode=args.mode)
   
    
    ddpm_model = DDPM(n_object_class=num_classes)
    optimizer = torch.optim.Adam(ddpm_model.net.parameters(), lr=args.lr)
    noise_scheduler = DDPMScheduler( num_train_timesteps=args.num_train_timesteps,
            beta_start=0.0001,
            beta_end=args.beta_end,
            beta_schedule='squaredcos_cap_v2'
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    if args.mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train(ddpm_model, loader, optimizer, noise_scheduler, scheduler, num_epochs=args.num_epoch)
    else:
        checkpoint = torch.load(args.checkpoint)
        ddpm_model.load_state_dict(checkpoint['model'])
        loader = DataLoader(dataset, batch_size=1)
        test(ddpm_model, noise_scheduler, loader, args.out_root)

