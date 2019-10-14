import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import grad

from tqdm.auto import trange, tqdm


import lib.collections as lc
from lib.utils import score_func, NoGrad
from lib.utils import compute_grad_norm, plot_hist


class MLPGenerator(nn.Module):
    def __init__(
            self,
            latent_dim,
            condition_dim,
            hidden_dims,
            target_dim,
            activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(activation)
            return nn.Sequential(*layers)

        self.net = nn.Sequential()
        self.net.add_module(
            'block_0',
            block(latent_dim + condition_dim, hidden_dims[0], False)
        )
        if len(hidden_dims) > 1:
            for i, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:]), 1):
                self.net.add_module(
                    f'block_{i}',
                    block(in_dim, out_dim)
                )
        self.net.add_module(
            'output', nn.Linear(hidden_dims[-1], target_dim)
        )

    def forward(self, Z, C):
        inputs = torch.cat([Z.squeeze(), C.squeeze()], dim=-1)
        return self.net(inputs)


class MLPDiscriminator(nn.Module):
    def __init__(
            self,
            target_dim,
            condition_dim,
            hidden_dims,
            output_dim=1,
            activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module(
            'fc_0',
            nn.Sequential(
                nn.Linear(target_dim + condition_dim, hidden_dims[0]),
                activation
            )

        )
        if len(hidden_dims) > 1:
            for i, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:]), 1):
                self.net.add_module(
                    f'fc_{i}',
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        activation,
                    )
                )
        self.net.add_module(
            'logits', nn.Linear(hidden_dims[-1], output_dim)
        )

    def forward(self, X, C):
        inputs = torch.cat([X.squeeze(), C.squeeze()], dim=-1)
        return self.net(inputs)


class GAN:

    def __init__(
            self,
            generator,
            discriminator,
            generator_opt,
            discriminator_opt,

    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_opt = generator_opt
        self.discriminator_opt = discriminator_opt
        assert next(self.generator.parameters()).device == next(self.discriminator.parameters()).device
        self.device = next(self.generator.parameters()).device

    @staticmethod
    def make_dataloader(targets, conditions, batch_size, num_workers):
        dataset = lc.Dataset(conditions, targets)
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader

    def test(self, dataloader, tqdm=None, plot_dists=False, n_slises=300):
        reference = []
        generated = []
        with torch.no_grad():
            for C, X in dataloader:
                reference.append(torch.cat([X, C], dim=-1).squeeze())
                Z = torch.randn(len(X), self.generator.latent_dim, dtype=torch.float, device=self.device)
                X_prime = self.generator(Z, C.to(self.device)).cpu()
                generated.append(torch.cat([X_prime, C.squeeze()], dim=-1))
                if tqdm is not None:
                    tqdm.update()

        reference = torch.cat(reference, dim=0).numpy()
        generated = torch.cat(generated, dim=0).numpy()
        if plot_dists:
            figs = list()
            for i in range(X.size(1)):
                figs.append(plot_hist(reference[:, i], generated[:, i]))
        else:
            figs = None

        return score_func(reference, generated, n_slises), figs

    def train(
            self,
            dataloaders,
            start_epoch=0,
            num_epochs=100,
            n_critic=5,
            writer=None,
            test_freq=1,
            log_grad_norms=False,
            plot_dists=False
    ):

        epoch_tqdm = trange(start_epoch, start_epoch + num_epochs)
        batch_tqdm = {
            phase: tqdm(total=len(dataloaders['train']), desc=f'{phase}')
            for phase in dataloaders
        }
        for epoch in epoch_tqdm:
            g_loss_sum = 0
            g_loss_count = 0
            d_loss_sum = 0
            d_loss_count = 0

            for i, (C, X) in enumerate(dataloaders['train']):



                X = X.float().to(self.device)
                C = C.float().to(self.device)

                d_loss, d_grad_norms = self.train_discriminator(
                    X,
                    C,
                    log_grad_norms,
                )
                d_loss_sum += d_loss * len(X)
                d_loss_count += len(X)
                step = epoch * len(dataloaders['train']) + i
                if log_grad_norms:
                    writer.add_scalar('grads/discriminator grad norms', d_grad_norms, global_step=step)

                if i % n_critic == 0:
                    g_loss, g_grad_norms = self.train_generator(
                        X,
                        C,
                        log_grad_norms
                    )
                    g_loss_sum += g_loss*len(X)
                    g_loss_count += len(X)
                    if log_grad_norms:
                        writer.add_scalar('grads/generator grad norms', g_grad_norms, global_step=step)

                batch_tqdm['train'].update()

            score = None
            figs =  None
            if 'test' in dataloaders and (epoch - start_epoch) % test_freq == 0:
                score, figs = self.test(dataloaders['test'], tqdm=batch_tqdm['test'], plot_dists=plot_dists)

            if writer is not None:
                writer.add_scalar('loss/generator loss', np.mean(g_loss_sum/g_loss_count), global_step=epoch)
                writer.add_scalar('loss/disciminator loss', np.mean(d_loss_sum/d_loss_count), global_step=epoch)
                if score is not None:
                    writer.add_scalar('KS-score', score, global_step=epoch)
                if figs is not None:
                    for i, fig in enumerate(figs):
                        writer.add_figure(f'column-{i}', fig, global_step=epoch)
            for t in batch_tqdm.values():
                t.reset()

    def generate(self, C):

        Z = torch.randn(len(C), self.generator.latent_dim, device=self.device, dtype=torch.float)
        C = torch.from_numpy(C).float().to(self.device)
        with torch.no_grad():
            X = self.generator(Z, C)
        return X.cpu().numpy()


class CGAN(GAN):

    def __init__(
            self,
            generator,
            discriminator,
            generator_opt,
            discriminator_opt,
            relativistic=None,
            smooth_labels=None,
            flip_labels=None,
            generator_loss=nn.BCEWithLogitsLoss(),
            discriminator_loss=nn.BCEWithLogitsLoss(),
    ):
        super().__init__(generator, discriminator, generator_opt, discriminator_opt)
        self.relativistic = relativistic
        self.smooth_labels = smooth_labels
        self.flip_labels = flip_labels
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def train_generator(self, real_batch, conditions, log_grad_norms):
        b_size = len(real_batch)
        real_labels = torch.ones(b_size, 1, dtype=torch.float, device=self.device)

        self.generator_opt.zero_grad()
        z = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
        fake_batch = self.generator(z, conditions)
        with NoGrad(self.discriminator):
            validity = self.discriminator(fake_batch, conditions)
        g_loss = self.generator_loss(validity, real_labels)
        g_loss.backward()
        self.generator_opt.step()
        if log_grad_norms:
            return g_loss.item(), compute_grad_norm(self.generator)
        else:
            return g_loss.item(), None

    def train_discriminator(self, real_batch, conditions, log_grad_norms):

        b_size = len(real_batch)
        real_labels = torch.ones(b_size, 1, dtype=torch.float, device=self.device)
        fake_labels = torch.zeros(b_size, 1, dtype=torch.float, device=self.device)
        if self.smooth_labels is not None:
            real_labels += self.smooth_labels[0] * (torch.rand_like(real_labels) - torch.rand_like(real_labels))
            fake_labels += self.smooth_labels[1] * torch.rand_like(fake_labels)

        if self.flip_labels is not None:
            real_mask = torch.from_numpy(np.random.binomial(1, 1 - self.flip_labels[0], len(real_labels))).float().to(
                self.device).view(-1, 1)
            fake_mask = torch.from_numpy(np.random.binomial(1, 1 - self.flip_labels[1], len(fake_labels))).float().to(
                self.device).view(-1, 1)
            new_real_labels = real_mask * real_labels + (1 - real_mask) * fake_labels
            new_fake_labels = fake_mask * fake_labels + (1 - fake_mask) * real_labels
            real_labels = new_real_labels
            fake_labels = new_fake_labels

        self.discriminator_opt.zero_grad()

        validity_real = self.discriminator(real_batch, conditions)
        with torch.no_grad():
            z = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
            fake_batch = self.generator(z, conditions)
        validity_fake = self.discriminator(fake_batch, conditions)
        if self.relativistic is not None:
            if self.relativistic == 'average':
                d_fake_loss = self.discriminator_loss(validity_fake - validity_real.mean(), fake_labels)
                d_real_loss = self.discriminator_loss(validity_real - validity_fake.mean(), real_labels)
            else:
                d_fake_loss = self.discriminator_loss(validity_fake - validity_real, fake_labels)
                d_real_loss = self.discriminator_loss(validity_real - validity_fake, real_labels)
        else:
            d_fake_loss = self.discriminator_loss(validity_fake, fake_labels)
            d_real_loss = self.discriminator_loss(validity_real, real_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.discriminator_opt.step()

        if log_grad_norms:
            return d_loss.item(), compute_grad_norm(self.discriminator)
        else:
            return d_loss.item(), None


class WGAN(GAN):

    def __init__(
            self,
            generator,
            discriminator,
            generator_opt,
            discriminator_opt,
            lambda_gp,
    ):
        super().__init__(generator, discriminator, generator_opt, discriminator_opt)
        self.lambda_gp = lambda_gp

    def _compute_gradient_penalty(self, real_batch, fake_batch, conditions):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(len(real_batch), 1).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_batch + ((1 - alpha) * fake_batch)).requires_grad_(True)
        validity_inter = self.discriminator(interpolates, conditions)
        labels = torch.ones(len(real_batch), 1, dtype=torch.float, device=self.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=validity_inter,
            inputs=interpolates,
            grad_outputs=labels,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(len(real_batch), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_generator(self, real_batch, conditions, log_grad_norms):
        self.generator_opt.zero_grad()
        b_size = len(real_batch)
        z = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
        fake_batch = self.generator(z, conditions)
        with NoGrad(self.discriminator):
            validity = self.discriminator(fake_batch, conditions)

        g_loss = - validity.mean()
        g_loss.backward()
        self.generator_opt.step()

        if log_grad_norms:
            return g_loss.item(), compute_grad_norm(self.generator)
        else:
            return g_loss.item(), None

    def train_discriminator(self, real_batch, conditions, log_grad_norms):
        b_size = len(real_batch)
        self.discriminator_opt.zero_grad()

        validity_real = self.discriminator(real_batch, conditions)
        with torch.no_grad():
            z = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
            fake_batch = self.generator(z, conditions)
        validity_fake = self.discriminator(fake_batch, conditions)

        gp = self._compute_gradient_penalty(real_batch, fake_batch, conditions)

        d_loss = -validity_real.mean() + validity_fake.mean() + self.lambda_gp*gp
        d_loss.backward()
        self.discriminator_opt.step()

        if log_grad_norms:
            return d_loss.item(), compute_grad_norm(self.discriminator)
        else:
            return d_loss.item(), None


class CramerGAN(GAN):

    def __init__(
            self,
            generator,
            discriminator,
            generator_opt,
            discriminator_opt,
            lambda_gp,
            surogate=False,
    ):
        super().__init__(generator, discriminator, generator_opt, discriminator_opt)
        self.lambda_gp = lambda_gp
        self.surogate = surogate

    def _critic(self, real_batch, fake_batch, conditions):

        validity_real = self.discriminator(real_batch, conditions)
        validity_fake = self.discriminator(fake_batch, conditions)

        f = torch.norm(validity_real - validity_fake, p=2, dim=-1) - \
            torch.norm(validity_real, p=2, dim=-1)
        return f

    def train_generator(self, real_batch, conditions, log_grad_norms):
        self.generator_opt.zero_grad()
        b_size = len(real_batch)
        z1 = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
        z2 = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
        fake_batch_1 = self.generator(z1, conditions)
        fake_batch_2 = self.generator(z2, conditions)
        with NoGrad(self.discriminator):
            c1 = self._critic(real_batch, fake_batch_2, conditions)
            c2 = self._critic(fake_batch_1, fake_batch_2, conditions)
        g_loss = torch.mean(c1 - c2)
        g_loss.backward()
        self.generator_opt.step()
        if log_grad_norms:
            return g_loss.item(), compute_grad_norm(self.generator)
        else:
            return g_loss.item(), None

    def train_discriminator(self, real_batch, conditions, log_grad_norms=False):
        self.discriminator_opt.zero_grad()
        b_size = len(real_batch)
        with torch.no_grad():
            z1 = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
            z2 = torch.randn(b_size, self.generator.latent_dim, dtype=torch.float, device=self.device)
            fake_batch_1 = self.generator(z1, conditions)
            fake_batch_2 = self.generator(z2, conditions)

        surrogate = self._critic(real_batch, fake_batch_2, conditions)\
                    - self._critic(fake_batch_1, fake_batch_2, conditions)

        d_loss = -surrogate.mean() + self.lambda_gp*self._compute_gradient_penalty(real_batch, fake_batch_1, conditions)
        d_loss.backward()
        self.discriminator_opt.step()

        if log_grad_norms:
            return d_loss.item(), compute_grad_norm(self.discriminator)
        else:
            return d_loss.item(), None

    def _compute_gradient_penalty(self, real_batch, fake_batch, conditions):
        b_size = len(real_batch)
        alpha = torch.rand(b_size, 1, dtype=torch.float, device=self.device)
        interpolates = alpha*real_batch + (1 - alpha)*fake_batch
        interpolates.requires_grad_(True)
        validity_int = self._critic(interpolates, fake_batch, conditions)
        gradients = grad(
            outputs=validity_int,
            inputs=interpolates,
            grad_outputs=torch.ones(b_size, dtype=torch.float, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gp = torch.mean((gradients.norm(2, dim=1) - 1)**2)
        return gp
