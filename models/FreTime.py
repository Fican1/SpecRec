import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.RWKV_7 import RWKV7Block

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)


        self.embed_size = self.seq_len
        self.hidden_size = configs.hidden_size
        self.e_layers = configs.e_layers

        self.rwkv_real = RWKV7Block(
            dim=self.embed_size//2 +1,
            block_id=0,
            n_blocks=1,
        )
        self.rwkv_imag = RWKV7Block(
            dim=self.embed_size//2 +1,
            block_id=0,
            n_blocks=1,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),  
            nn.LeakyReLU(), 
            nn.Linear(self.hidden_size, self.pred_len)  
        )


    def fft_rwkv(self, x):
        x = torch.fft.rfft(x, dim=2, norm='ortho')

        x_real = x.real
        x_imag = x.imag

        x_real,_=self.rwkv_real(x_real)
        x_imag,_=self.rwkv_imag(x_imag)

        y=torch.complex(x_real,x_imag)


        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z
        x = x.permute(0, 2, 1)

        x = self.fft_rwkv(x)  

        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        return x
