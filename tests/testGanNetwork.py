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
        cls.test_df = cls.test_df.drop(['id'], axis=1)

    def test_check_for_model(self):

        pz_gan = ganNetwork()
        err_msg = "No model trained. Run train_gan method first."
        with self.assertRaises(ValueError, msg=err_msg):
            pz_gan.create_gan_cat(10, 10)

    def test_train_gan(self):

        pz_gan = ganNetwork()
        n_epochs = 10
        pz_gan.train_gan(self.test_df.values, n_epochs)

        z_noise = Variable(torch.randn(len(self.test_df),
                           np.shape(self.test_df)[1]))
        gan_output = pz_gan.gan_model(z_noise)
        new_sample = gan_output.detach().numpy() * \
                     pz_gan.data_std + pz_gan.data_mean

        np.testing.assert_array_equal(np.shape(self.test_df.values),
                                      np.shape(new_sample))

    def test_create_gan_cat(self):

        pz_gan = ganNetwork()
        n_epochs = 10
        pz_gan.train_gan(self.test_df.values, n_epochs)
        n_columns = np.shape(self.test_df.values)[1]

        cat_length = 100
        new_cat = pz_gan.create_gan_cat(cat_length)
        np.testing.assert_array_equal((cat_length, n_columns),
                                      np.shape(new_cat))

        new_cat_2, input_noise = pz_gan.create_gan_cat(cat_length, True)
        np.testing.assert_array_equal((cat_length, n_columns),
                                      np.shape(new_cat_2))
        np.testing.assert_array_equal((cat_length, n_columns),
                                      np.shape(input_noise))
                                    
    def test_save_load_model(self):

        pz_gan = ganNetwork()
        n_epochs = 10
        pz_gan.train_gan(self.test_df.values, n_epochs)
        n_columns = np.shape(self.test_df.values)[1]

        pz_gan.save_model('test_out.model')

        model = ganNetwork()
        model.load_model('test_out.model', n_columns)
        np.testing.assert_array_equal(pz_gan.data_mean,
                                      model.data_mean)
        np.testing.assert_array_equal(pz_gan.data_std,
                                      model.data_std)

        cat_length = 100
        new_cat = model.create_gan_cat(cat_length)
        np.testing.assert_array_equal((cat_length, n_columns),
                                      np.shape(new_cat)) 

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('test_out.model'):
            os.remove('test_out.model')

if __name__ == '__main__':
    unittest.main()