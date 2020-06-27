import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd.variable import Variable
from copy import deepcopy

all = ['gan']


class gan():

    def __init__(self, seed=None):

        if seed is not None:
            torch.manual_seed(seed)

        self.disc = None
        self.gen = None
        self.gan_model = None
        self.data_mean = None
        self.data_std = None

    def set_training_models(self, discriminator, generator):

        self.disc = discriminator
        self.gen = generator

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.disc.cuda()
            self.gen.cuda()

    def check_for_model(self):

        if self.gan_model is None:
            raise ValueError("No model trained. Run train_gan method first.")

    def reset_grad(self, params):
        for p in params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())

    def train_gan(self, train_input, n_epochs,
                  mini_batch_size=512, disc_lr=0.0002, gen_lr=0.0002,
                  update_progress_freq=5):

        if ((self.disc is None) or (self.gen is None)):
            raise ValueError("No training models set. Use set_training_models.")

        # Code based off of work at https://github.com/wiseodd/generative-models

        Z_dim = np.shape(train_input)[1]

        self.disc.train()
        self.gen.train()

        d_optimizer = optim.Adam(self.disc.parameters(), lr=disc_lr)
        g_optimizer = optim.Adam(self.gen.parameters(), lr=gen_lr)

        # Prepare the data
        good_data = deepcopy(train_input)
        data_mean = np.mean(good_data, axis=0)
        data_std = np.std(good_data, axis=0)
        good_data -= data_mean
        good_data /= data_std
        input_data = torch.tensor(good_data, dtype=torch.float)

        d_params = [x for x in self.disc.parameters()]
        g_params = [x for x in self.gen.parameters()]

        params = d_params + g_params

        ones_label = Variable(torch.ones(mini_batch_size, 1))
        zeros_label = Variable(torch.zeros(mini_batch_size, 1))

        epoch_samples = []

        for epoch_num in range(n_epochs):

            num_batches = np.ceil(len(input_data)/mini_batch_size)
            input_idx = np.arange(len(input_data))
            np.random.shuffle(input_idx)
            for it in range(int(num_batches)):
                # Sample data
                X = input_data[input_idx[it*mini_batch_size:
                                         (it+1)*mini_batch_size]]
                z = Variable(torch.randn(len(X), Z_dim))
                X = Variable(X)

                ones_label = Variable(torch.ones(len(X), 1))
                zeros_label = Variable(torch.zeros(len(X), 1))

                # Update discriminator

                G_sample = self.gen(z)
                D_real = self.disc(X)
                D_fake = self.disc(G_sample)

                D_loss_real = F.binary_cross_entropy(D_real, ones_label)
                D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label)
                D_loss = D_loss_real + D_loss_fake

                D_loss.backward()
                d_optimizer.step()

                self.reset_grad(params)

                # Update generator

                z = Variable(torch.randn(mini_batch_size, Z_dim))
                G_sample = self.gen(z)
                D_fake = self.disc(G_sample)

                ones_label = Variable(torch.ones(len(z), 1))
                zeros_label = Variable(torch.zeros(len(z), 1))

                G_loss = F.binary_cross_entropy(D_fake, ones_label)

                G_loss.backward()
                g_optimizer.step()

                self.reset_grad(params)

            if update_progress_freq == 0:
                continue
            else:
                if epoch_num % update_progress_freq == 0:
                    print('Epoch: %i, Iter: %i, D_loss: %.3f, G_loss: %.3f' %
                            (epoch_num, it, D_loss, G_loss))

            z = Variable(torch.randn(len(train_input), Z_dim))
            G_sample = self.gen(z)
            new_sample = G_sample.detach().numpy() * data_std + data_mean

        self.gan_model = deepcopy(self.gen)
        self.data_mean = data_mean
        self.data_std = data_std

        return

    def create_gan_data(self, cat_length, return_input=False):

        self.check_for_model()

        self.gan_model.eval()
        n_columns = len(self.data_mean)
        z_noise = Variable(torch.randn(cat_length, n_columns))
        G_sample = self.gan_model(z_noise)
        new_sample = G_sample.detach().numpy() * self.data_std + self.data_mean

        if return_input is False:
            return new_sample
        else:
            return new_sample, z_noise.numpy()

    def save_model(self, filename):

        self.check_for_model()

        torch.save({'model_state_dict': self.gan_model.state_dict(),
                    'model_train_mean': self.data_mean,
                    'model_train_stdev': self.data_std},
                   filename)

    def load_model(self, filename, n_dim):

        g = generator(n_dim, n_dim*4)
        checkpoint = torch.load(filename)
        g.load_state_dict(checkpoint['model_state_dict'])
        g.train_mean = checkpoint['model_train_mean']
        g.train_stdev = checkpoint['model_train_stdev']

        self.gan_model = g
        self.data_mean = g.train_mean
        self.data_std = g.train_stdev
