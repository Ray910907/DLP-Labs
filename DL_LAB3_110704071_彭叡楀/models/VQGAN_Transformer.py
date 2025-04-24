import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import matplotlib.pyplot as plt


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        qz, id, _ = self.vqgan.encode(x)
        return id.reshape(qz.shape[0], -1)
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.
        
        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        
        def linear_decay(ratio):
            return 1 - ratio

        def cosine_decay(ratio):
            return np.cos(ratio * np.pi / 2)

        def square_decay(ratio):
            return 1 - ratio * ratio

        if mode == "linear":
            return linear_decay
        elif mode == "cosine":
            return cosine_decay
        elif mode == "square":
            return square_decay
        else:
            raise ValueError("mode must be 'linear', 'cosine' ,'square'.")

##TODO2 step1-3:            
    def forward(self, x):
        #Get latent code of picture
        z_indices = self.encode_to_z(x)
        
        #Get the masked picture to training
        mask = torch.full(z_indices.shape, self.mask_token_id, dtype=torch.long).to(z_indices.device)

        mask_pos = torch.randint(0, 2, z_indices.shape, dtype=torch.bool).to(z_indices.device)
        mask[mask_pos == 0] = z_indices[mask_pos == 0]
        #Predict probability token
        logits = self.transformer(mask)

        return logits, z_indices

    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self,ratio,mask_func,mask_num,mask_b,z_indices):
        z_masked = torch.full(z_indices.shape, self.mask_token_id, dtype=torch.long).to(z_indices.device)
        z_masked[mask_b == 0] = z_indices[mask_b == 0]
        logits = self.transformer(z_masked)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.softmax(logits,dim=-1)

        #FIND MAX probability for each token value

        z_indices_predict_prob, z_indices_predict = torch.max(logits,dim=-1)   
        z_indices_predict[mask_b == 0] = z_indices[mask_b == 0]

        self.gamma = self.gamma_func(mode = mask_func)
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(logits.device)  # gumbel noise
        temperature = self.choice_temperature * (1 - self.gamma(ratio))
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        confidence[mask_b == 0] = torch.inf
        
        remain = int(mask_num * self.gamma(ratio) // 1)
        paint, _ = confidence.topk(remain, dim=-1, largest=False)

        threshold = paint[:, -1] if remain > 0 else torch.full((confidence.shape[0],), confidence.min().item(), device=confidence.device)
        mask_bc = confidence < threshold

        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
