import torch 
import torch.nn as nn
from torch.nn.modules import Module
from torch.autograd import Variable


class CVAE(Module):
    def __init__(self, encoder_dims, latent_dim, decoder_dims,cond_dim):
        super().__init__()

        assert isinstance(encoder_dims, list)
        assert isinstance(decoder_dims, list)
        assert isinstance(latent_dim, int)
        assert isinstance(cond_dim, int)

        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.encoder = Encoder(encoder_dims, latent_dim)
        self.decoder = Decoder(decoder_dims, latent_dim,cond_dim)

    def forward(self, x, c):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5*log_var)
        eps = torch.randn(batch_size, self.latent_dim).to(std.device)

        z = eps*std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var

    def infer(self, c, n=1):

        z = Variable(torch.randn(n, self.latent_dim))
        
        recon_x = self.decoder(z, c)
        
        return recon_x


class Encoder(Module):

    def __init__(self, layer_dims, latent_dim):

        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(in_size,out_size))
            self.MLP.add_module(name=f'A{i}', module=nn.ReLU())

        self.linear_means = nn.Linear(layer_dims[-1], latent_dim)
        self.linear_logvar = nn.Linear(layer_dims[-1], latent_dim)
    
    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)
        means = self.linear_means(x)
        log_var = self.linear_logvar(x)

        return means, log_var


class Decoder(Module):
    
    def __init__(self, layer_dims, latent_dim, cond_dim):

        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_dim + cond_dim

        for i, (in_size, out_size) in enumerate( zip([input_size]+layer_dims[:-1], layer_dims)):
            self.MLP.add_module(name=f'L{i}', module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_dims):
                self.MLP.add_module(name=f'A{i}', module=nn.ReLU())
            # else:
            #     self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):
        
        z = torch.cat((z, c), dim=-1)

        return self.MLP(z)


