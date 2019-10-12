import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

from models.gans import CGAN, MLPDiscriminator, MLPGenerator

latent_dim = 20
condition_dim = 3
d_hidden_dims = [32, 64, 128]
g_hidden_dims = [32, 64, 128]
target_dim = 5
seed = 42

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
condition_transformer = make_pipeline(QuantileTransformer(random_state=seed), MinMaxScaler((-1, 1)))
target_transformer = make_pipeline(QuantileTransformer(random_state=seed), MinMaxScaler((-1, 1)))

config = dict(
    dataloader={'batch_size': 128, 'num_workers':4},
    train=dict(
    start_epoch=0,
    num_epochs=20,
    n_critic=5,
)
)
