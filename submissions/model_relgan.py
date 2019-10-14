import torch
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

from models.gans import CGAN, MLPDiscriminator, MLPGenerator

seed = 52
latent_dim = 50
condition_dim = 3
d_hidden_dims = [32, 64, 128, 128]
g_hidden_dims = [32, 64, 128, 128]
target_dim = 5

device = torch.device('cuda:0')
generator = MLPGenerator(latent_dim, condition_dim, g_hidden_dims, target_dim,).to(device)
discriminator = MLPDiscriminator(target_dim, condition_dim, d_hidden_dims).to(device)

generator_opt = optim.Adam(generator.parameters(),  lr=1e-4, betas=(0, 0.9))
discriminator_opt = optim.Adam(discriminator.parameters(),  lr=1e-4, betas=(0, 0.9))

model = CGAN(
    generator,
    discriminator,
    generator_opt,
    discriminator_opt,
    relativistic=None,
    smooth_labels=(0.1, 0.1),
    flip_labels=None,
)

condition_transformer = QuantileTransformer(output_distribution='normal', random_state=seed)
target_transformer = QuantileTransformer(output_distribution='normal', random_state=seed)

config = dict(
    dataloader={'batch_size': 256, 'num_workers': 4},
    train=dict(
        start_epoch=0,
        num_epochs=20,
        n_critic=5,
    )
)
