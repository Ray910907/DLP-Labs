import os
import argparse
import numpy as np
import torch
import math
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        
        self.ep = current_epoch
        self.all_ep = args.num_epoch

        self.kl_type = args.kl_anneal_type
        self.beta = 0.015 if(self.kl_type != 'None') else 1.0

        self.kl_cycle = args.kl_anneal_cycle
        self.kl_ratio = args.kl_anneal_ratio
        
        
    def update(self):
        # TODO
        if(self.kl_type != 'None'):
            self.frange_cycle_linear(n_iter=self.all_ep, ratio=self.kl_ratio)
        self.ep += 1
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.015, stop=1.0, n_cycle=1, ratio=1):
        # TODO
        if (self.kl_type == 'Cyclical'):
            n_cycle = self.kl_cycle
        period = n_iter // n_cycle #period per cycle
        anneal_climb = period * ratio #ratio of annealing phase
        interval = (stop - start) / (math.ceil(anneal_climb) - 1) if(math.ceil(anneal_climb) != 1) else 0.0
        
        if(self.ep % period < anneal_climb):
            self.beta = min(start + (self.ep % period) * interval,stop)
            

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.device = args.device
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.state_initial = {k: v.clone().detach() for k, v in self.state_dict().items()}
        self.loss_val = float('inf')
        self.plot_psnr = args.plot_psnr
        self.plot_training_loss = args.plot_training_loss
        self.no_weight_interpolation = args.no_weight_interpolation
    
    def weight_interpolation(self):
        #current and original weight
        weights_after = {}
        weights_now = self.state_dict()
        #check whether the weight is legal, then doing cosine interpolation
        for i in weights_now:
            w_12 = weights_now[i].flatten().float()
            w_0 = self.state_initial[i].flatten().float().to(self.device)

            if i not in self.state_initial or w_12.shape != w_0.shape:
                weights_after[i] = weights_now[i]

            else:
                cosine = torch.nn.functional.cosine_similarity(w_12, w_0, dim=0).clamp(-1 + 1e-7, 1 - 1e-7)
                alpha = (2 * cosine) / (1 + cosine)
                interpolated = alpha * w_12 + (1 - alpha) * w_0
                weights_after[i] = interpolated.view_as(weights_now[i])

        self.load_state_dict(weights_after)

    def forward(self, img, label):
        pass
    
    def training_stage(self):
        losses = []
        losses_id = []
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, round(beta,3)), pbar, loss.detach().cpu(), lr=round(self.scheduler.get_last_lr()[0], 8))
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, round(beta,3)), pbar, loss.detach().cpu(), lr=round(self.scheduler.get_last_lr()[0], 8))
            if(self.plot_training_loss):
                losses.append(loss.item())
                losses_id.append(i+1)

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            loss = self.eval()

            if self.loss_val > loss:
                self.loss_val = loss
                self.save(os.path.join(self.args.save_root, f"val.ckpt"))
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

            if ~self.no_weight_interpolation and self.current_epoch >= 5 and self.current_epoch % 5 == 0:
                self.weight_interpolation()
        
        if(self.plot_training_loss):
                plt.clf()
            
                plt.plot(losses_id, losses)
                plt.title(f"Train_loss")
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.show()
                plt.savefig(f'output/train_loss.png')
                plt.close()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        loss_total = 0.0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss = self.val_one_step(img, label)
            loss_total += loss
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        return loss_total
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        img_pred = img[:, 0]
        beta = self.kl_annealing.get_beta()
        loss_total = 0.0

        for step in range(1, self.train_vi_len):
            #choose the reference image base on whether use the teacher forcing strategy
            img_ref = img[:, step - 1] * self.tfr + img_pred * (1-self.tfr) if adapt_TeacherForcing else img_pred.detach()
            #get through the encoder to get more features 
            img_in = self.frame_transformation(img_ref)
            label_in = self.label_transformation(label[:, step])
            gt_in = self.frame_transformation(img[:, step])
            #use the features get the mean and variance of q(z|x; theta) then merge all features to generate the predict image
            z, mu, logvar = self.Gaussian_Predictor(gt_in,label_in)
            fuse = self.Decoder_Fusion(img_in,label_in,z)

            img_pred = torch.sigmoid(self.Generator(fuse))
            #calculate the loss
            mse_loss = self.mse_criterion(img_pred, img[:, step])
            kld_loss = kl_criterion(mu, logvar, self.batch_size)
            loss = mse_loss + beta * kld_loss
            loss_total += loss
        #backpropogation
        loss_total.backward()
        self.optimizer_step()
        self.optim.zero_grad()
        return loss_total / (self.train_vi_len - 1)
    
    def val_one_step(self, img, label):
        # TODO
        img_ref = img[:, 0]
        loss_total = 0.0
        psnrs = []
        psnrs_id = []
        for step in range(1, self.val_vi_len):
            img_in = self.frame_transformation(img_ref)
            label_in = self.label_transformation(label[:, step])

            z = torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)
            fuse = self.Decoder_Fusion(img_in,label_in,z)

            img_pred = self.Generator(fuse)
            img_ref = torch.sigmoid(img_pred)

            mse_loss = self.mse_criterion(img_ref, img[:, step])
            loss_total += mse_loss

            if(self.plot_psnr):
                psnr = Generate_PSNR(img[:,step], img_ref)
                psnrs.append(psnr.item())
                psnrs_id.append(step)
        
        if(self.plot_psnr):
            plt.clf()
            
            plt.plot(psnrs_id, psnrs)
            plt.title("Valid_PSNR")
            plt.xlabel('frame')
            plt.ylabel('psnr')
            plt.show()
            plt.savefig(f'output/PSNR.png')
            plt.close()
            
        return loss_total / (self.val_vi_len - 1)
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        self.tfr = max(self.tfr - self.tfr_d_step,0.0) if (self.current_epoch >= self.tfr_sde) else self.tfr
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    parser.add_argument('--plot_psnr',     action='store_true')
    parser.add_argument('--plot_training_loss',     action='store_true')
    parser.add_argument('--no_weight_interpolation',     action='store_true')

    args = parser.parse_args()
    
    main(args)
