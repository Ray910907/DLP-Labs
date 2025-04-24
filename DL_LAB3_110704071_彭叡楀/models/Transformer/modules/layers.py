import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = num_heads
        self.dim = dim
        
        if(dim % num_heads):
            raise ValueError("Error!")
        
        self.LQ = nn.Linear(dim,dim)
        self.LV = nn.Linear(dim,dim)
        self.LK = nn.Linear(dim,dim)

        self.drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(dim, dim)
        
    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size = x.shape[0]
        num_tokens = x.shape[1]

        Q = self.LQ(x)
        Q = Q.view(batch_size, num_tokens, self.head, self.dim // self.head).transpose(1, 2)

        V = self.LV(x)
        V = V.view(batch_size, num_tokens, self.head, self.dim // self.head).transpose(1, 2)

        K = self.LK(x)
        K = K.view(batch_size, num_tokens, self.head, self.dim // self.head).transpose(1, 2)
        
        #calculate self attention
        relation = torch.matmul(Q, K.transpose(-2, -1))
        soft = torch.softmax(relation / (self.dim ** 0.5), dim=-1)
        attention = self.drop(soft)

        #apply attention weights to V
        out = torch.matmul(attention, V)

        #concatenate heads and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.dim)

        #final linear projection
        return self.projection(out)

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    