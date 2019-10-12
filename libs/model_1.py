import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.gans import CGAN, MLPDiscriminator, MLPGenerator

latent_dim = 20
condition_dim = 3
d_hidden_dims = [32, 64, 128]
g_hidden_dims = [32, 64, 128]
target_dim = 5

device = torch.device('cuda:0')
generator = MLPGenerator(latent_dim, condition_dim, g_hidden_dims, target_dim,).to(device)
discriminator = MLPDiscriminator(target_dim, condition_dim, d_hidden_dims).to(device)

generator_opt = optim.Adam(generator.parameters())
discriminator_opt = optim.Adam(discriminator.parameters())

model = CGAN(
    generator,
    discriminator,
    generator_opt,
    discriminator_opt,
    relativistic=True,
    flip_labels=(0.01, 0.01),
    smooth_labels=(0.1, 0.1)
)
condition_transformer = None
target_transformer = None
config = dict(
    start_epoch=0,
    num_epochs=20,
    device=torch.device('cuda:0'),
    n_critic=1,
    writer=SummaryWriter(log_dir='/_data/richgan/runs/model_1'),
    test_freq=1,
)
