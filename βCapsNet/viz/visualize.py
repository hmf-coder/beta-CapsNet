import numpy as np
import torch
from viz.latent_traversals import LatentTraverser
from scipy import stats
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image


class Visualizer():
    def __init__(self, model):
        """
        Generate images of samples, reconstructions, latent traversals and so on.

        model : capsnet instance
        """
        self.model = model
        self.latent_traverser = LatentTraverser()
        self.save_images = False  # If false, each method returns a tensor instead of saving image.

    def reconstructions(self, data, labels, size=(8, 8), filename='recon.png'):
        """
        Generates reconstructions of data through model.reconstruction_net

        data : torch.Tensor  Data to be reconstructed. Shape (Batch, C, H, W)
        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()    # before test.      fix params of trained model  权重不受batchnorm的影响
        # Pass data through capsnet to obtain reconstruction
        with torch.no_grad():
            input_data = data
            input_labels = labels
        recon_data, _, _ = self.model(input_data, input_labels)
        self.model.train()    # 获取重构样本

        # Upper half of plot will contain data, bottom half will contain reconstructions
        num_images = int(size[0] * size[1] / 2)
        originals = input_data[:num_images].cpu()
        reconstructions = recon_data.view(-1, data.size()[1], data.size()[2], data.size()[3])[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]   
        if num_images > num_examples:   # input_data[:num_images]超出索引范围也不会报错，但input_data[num_images]超出索引会报错
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        if self.save_images:
            save_image(comparison.data, filename, nrow=size[0])
        else:
            return make_grid(comparison.data, nrow=size[0])   # make_grid返回值是tensor

    def samples(self, class_caps, labels, size=(16, 10), filename='samples.png', range_=(-0.25,0.25), traverser=10):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        sample = class_caps.detach().numpy().squeeze(0)

        n_classes = sample.shape[0]    #class num
        n_dim = sample.shape[1]     #dim of each class capsule

        cdf_traversal = np.linspace(range_[0], range_[1], traverser)
        samples = []
        for i in range(n_dim):
            for j in range(n_classes):
                sample_copy = sample.copy()
                sample_copy[labels, i] = cdf_traversal[j]
                samples.append(sample_copy)
        #generated = self._reconstruction_net(prior_samples, labels)  #160,748
        #generate = generated.view(-1, 1, 28, 28)
            #save_image(generate.data, filename, nrow=size[1])
            #return make_grid(generate.data, nrow=size[1])
        return torch.Tensor(samples).view(n_dim*n_classes, n_dim*n_classes)
    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8,
                              filename='traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)   # make_grid: make a grid of images.  nrow: num of images in each row

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def all_latent_traversals(self, size=8, filename='all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):        # range(10)
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))     
        # 10 * (10 * 20)
        for disc_idx in range(self.model.num_disc_latents):     # num_disc_latents = 1
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,   # disc_idx = 0
                                                                      size=size))
        # 1 * (10 * 20)
        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))    # decoder((110 * 20)) 11个维度，每个遍历10遍，110 rows in total

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def _reconstruction_net(self, latent_samples, labels):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        return self.model.reconstruction_net(latent_samples, labels).cpu()


def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
    """
    Reorders rows or columns of an image grid.

    Parameters
    ----------
    orig_img : torch.Tensor
        Original image. Shape (channels, width, height)

    reorder : list of ints
        List corresponding to desired permutation of rows or columns

    by_row : bool
        If True reorders rows, otherwise reorders columns

    img_size : tuple of ints
        Image size following pytorch convention

    padding : int
        Number of pixels used to pad in torchvision.utils.make_grid
    """
    reordered_img = torch.zeros(orig_img.size())
    _, height, width = img_size

    for new_idx, old_idx in enumerate(reorder):
        if by_row:
            start_pix_new = new_idx * (padding + height) + padding
            start_pix_old = old_idx * (padding + height) + padding
            reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
        else:
            start_pix_new = new_idx * (padding + width) + padding
            start_pix_old = old_idx * (padding + width) + padding
            reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

    return reordered_img