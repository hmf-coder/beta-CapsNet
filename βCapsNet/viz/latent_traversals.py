import numpy as np
import torch
from scipy import stats


class LatentTraverser():
    def __init__(self):        
        """
        LatentTraverser is used to generate traversals of the latent space.

        Parameters
        ----------
        latent_spec : dict
            See jointvae.models.VAE for parameter definition.
        """
        #self.latent_spec = latent_spec    # latent_spec = {'cont': 10, 'disc': [10]}
        self.sample_prior = False  # If False fixes samples in untraversed latent dimensions. 
                                   # If True samples untraversed latent dimensions from prior.
        self.cont_dim = 16  # cont_dim = 10

    def traverse_line(self, cont_idx=None, disc_idx=None, size=5):
        """
        Returns a (size, D) latent sample, corresponding to a traversal of the
        latent variable indicated by cont_idx or disc_idx.

        Parameters
        ----------
        cont_idx : int or None
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and cont_idx = 7, then the 7th dimension
            will be traversed while all others will either be fixed or randomly
            sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        disc_idx : int or None
            Index of discrete latent dimension to traverse. If there are 5
            discrete latent variables and disc_idx = 3, then only the 3rd
            discrete latent will be traversed while others will be fixed or
            randomly sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        size : int
            Number of samples to generate.
        """
        samples = []

        samples.append(self._traverse_continuous_line(idx=cont_idx, size=size))

        return samples   # 10*20

    def _traverse_continuous_line(self, idx, size):
        """
        Returns a (size, cont_dim) latent sample, corresponding to a traversal
        of a continuous latent variable indicated by idx.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        size : int
            Number of samples to generate.
        """
        if self.sample_prior:
            samples = np.random.normal(size=(size, self.cont_dim))
        else:
            samples = np.zeros(shape=(size, self.cont_dim))      # zeros(10, 10)

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            cdf_traversal = np.linspace(0.05, 0.95, size)   # [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95]
            cont_traversal = stats.norm.ppf(cdf_traversal)  # Return the value of percentage point function. 对应概率的横坐标值
            # [-1.64485363 -1.03643339 -0.67448975 -0.38532047 -0.12566135  0.12566135 0.38532047  0.67448975  1.03643339  1.64485363]
            for i in range(size):
                samples[i, idx] = cont_traversal[i]

        return torch.Tensor(samples)

    def _traverse_discrete_line(self, dim, traverse, size):
        """
        Returns a (size, dim) latent sample, corresponding to a traversal of a
        discrete latent variable.

        Parameters
        ----------
        dim : int
            Number of categories of discrete latent variable.

        traverse : bool
            If True, traverse the categorical variable; if False, keep it fixed or randomly sample.

        size : int
            Number of samples to generate.
        """
        samples = np.zeros((size, dim))

        if traverse:
            for i in range(size):
                samples[i, i % dim] = 1.        # 对角线为1，其余为0的矩阵
        else:
            # Randomly select discrete variable (i.e. sample from uniform prior)
            if self.sample_prior:
                samples[np.arange(size), np.random.randint(0, dim, size)] = 1.
            else:
                samples[:, 0] = 1.             # 第一列为1，其余为0的矩阵

        return torch.Tensor(samples)   # 10*10

    def traverse_continuous_grid(self, class_caps, labels, size, axis=0, n_class=10, digit_num=0, range_=(-0.5,0.5), traverser=10): 
        """
        Returns a (size[0] * size[1], cont_dim) latent sample, corresponding to
        a two dimensional traversal of the continuous latent space.

        Parameters
        ----------
        axis : int
            Either 0 for traversal across the rows or 1 for traversal across
            the columns.

        size : tuple of ints
            Shape of grid to generate. E.g. (6, 4).
        """
        samples = class_caps.detach().numpy().squeeze(0)  
        # class_caps is torch.Size([1, 10, 16]), samples is numpy (10,16)
        cdf_traversal = np.linspace(range_[0], range_[1], size[axis])
        for i in range(size[0]):
            for j in range(size[1]):
                samples[i*size[1] + j, i + digit_num*16] = cdf_traversal[j]

        return torch.Tensor(samples)  #(size0*size1, cont_dim*n_class) 160,160

    def _traverse_discrete_grid(self, dim, axis, traverse, size):
        """                           dim=10, axis=0, traverse=False, size=(8,8)
        Returns a (size[0] * size[1], dim) latent sample, corresponding to a
        two dimensional traversal of a discrete latent variable, where the
        dimension of the traversal is determined by axis.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        axis : int
            Either 0 for traversal across the rows or 1 for traversal across
            the columns.

        traverse : bool
            If True, traverse the categorical variable otherwise keep it fixed
            or randomly sample.

        size : tuple of ints
            Shape of grid to generate. E.g. (6, 4).
        """
        num_samples = size[0] * size[1]
        samples = np.zeros((num_samples, dim))

        if traverse:
            disc_traversal = [i % dim for i in range(size[axis])]
            for i in range(size[0]):
                for j in range(size[1]):
                    if axis == 0:
                        samples[i * size[1] + j, disc_traversal[i]] = 1.
                    else:
                        samples[i * size[1] + j, disc_traversal[j]] = 1.
        else:
            # Randomly select discrete variable (i.e. sample from uniform prior)
            if self.sample_prior:
                samples[np.arange(num_samples), np.random.randint(0, dim, num_samples)] = 1.
            else:
                samples[:, 0] = 1.

        return torch.Tensor(samples)