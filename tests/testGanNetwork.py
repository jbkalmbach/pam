import sys
sys.path.append('..')
import os
import unittest
import h5py
import pandas as pd
import numpy as np
import torch
from torch.autograd.variable import Variable
from pam.ganNetwork import ganNetwork

class testGanNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        f = h5py.File(os.path.join(os.path.dirname(__file__),
                                   'test_data', 'testdata.hdf5'))
        cls.test_df = pd.DataFrame(np.array([f['photometry'][k][()] \
                                            for k in f['photometry'].keys()]).T,
                                   columns=f['photometry'].keys())

    def test_train_gan(self):

        print(len(self.test_df))

        pz_gan = ganNetwork()
        pz_gan.train_gan(self.test_df.values[:, 1:], 10)

        z_noise = Variable(torch.randn(10, 13))
        gan_output = pz_gan.gan_model(z_noise)
        new_sample = gan_output.detach().numpy() * \
                     pz_gan.data_std + pz_gan.data_mean
        print(new_sample)
        return

if __name__ == '__main__':
    unittest.main()