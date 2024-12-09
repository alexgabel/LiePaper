from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

import matplotlib.pyplot as plt


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

def get_coords(image, b=(0,0), r=1.0):
    # Get the size of the image
    N = image.shape[0]
    M = image.shape[1]
    # Create a meshgrid
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,M)
    X,Y = np.meshgrid(x,y)

    # For each point in the meshgrid, invert by dividing by 
    # the norm squared:
    sq_norm = (X**2 + Y**2)
    X_inv = r**2*X/sq_norm
    Y_inv = r**2*Y/sq_norm
    # Subtract vector b to the inversion coordinates
    # X_inv_b = X_inv - b[0]
    # Y_inv_b = Y_inv - b[1]
    # Must flip one of the components because of image coordinates
    X_inv_bp = X_inv + b[0]
    Y_inv_bp = Y_inv - b[1]

    sqp_norm = (X_inv_bp**2 + Y_inv_bp**2)
    Xp_new = r**2*X_inv_bp/sqp_norm
    Yp_new = r**2*Y_inv_bp/sqp_norm

    return X, Y, Xp_new, Yp_new


def SCT(image, b=[0,0], r=1.0, plot=False):
    """ This function applied a special conformal transformation to an image."""
    # Get coordinates of transformed grid
    X, Y, Xp_new, Yp_new = get_coords(image, b, r)

    original_coords = torch.stack([torch.tensor(X),torch.tensor(Y)],dim=-1)
    new_coords = torch.stack([torch.tensor(Xp_new),torch.tensor(Yp_new)],
                             dim=-1).reshape(*original_coords.shape)
    new_coords = new_coords.to(torch.float32)

    new_image = torch.nn.functional.grid_sample(image[None, None, ...], 
                                                new_coords[None, ...], 
                                                mode='bilinear', 
                                                padding_mode='zeros')

    return new_image

class SyMNIST(Dataset):
    def __init__(self, data_dir, tf_range=(0, 0, 0, 1, 0, 0, 0), dataset='mnist', pre_transform=None, post_transform=None, **kwargs):
        if dataset == 'mnist':    
            self.data = datasets.MNIST(data_dir, download=True, transform=pre_transform, **kwargs)
            self.n = self.data.data.shape[1]
        elif dataset == 'cifar10_g':
            self.data = datasets.CIFAR10(data_dir, download=True, transform=pre_transform, **kwargs)
            # Convert the CIFAR10 images to greyscale
            self.data.data = self.data.data.mean(axis=3)/255.0
            self.n = self.data.data.shape[1]

            # Reduce the size of the dataset for testing purposes:
            # self.data = Subset(self.data, range(128))
        # n is the size of the image (H)
        
        self.n = self.data.data.shape[1]
        
        print(tf_range)
        self.tx_max, self.ty_max, self.r_max, self.s_max, \
            self.sh_max, self.bx_max, self.by_max = tf_range
        self.transforms = post_transform

        # Initialize arrays to hold images and targets
        self.images = torch.zeros(len(self.data), self.n, self.n)
        self.targets = torch.zeros(len(self.data), self.n, self.n)

        for i, (image, _) in enumerate(self.data):
            # Angle is a random float between angle_range[0] and angle_range[1]
            # sign = np.random.choice([-1, 1])

            # Sample r from sum of two normals with mean self.r_max or -self.r_max and std 5.0
            r = np.random.uniform(-self.r_max, self.r_max)
            # Multilpy r by sign which is +1 or -1, randomy smpled
            # r = np.random.choice([-1, 1])*r
            #np.random.uniform(-self.r_max, self.r_max)#self.angle_range[0], self.angle_range[1])
            sh = np.random.uniform(-self.sh_max, self.sh_max)
            bx = np.random.uniform(-self.bx_max, self.bx_max)
            by = np.random.uniform(-self.by_max, self.by_max)
            # Angle_2 is a random float sampled from a trimodal distribution
            # with peaks at 0, 45 and -45 degrees and std of 0 degrees:
            # r = float(np.random.choice([self.r_max, -self.r_max]))

            # Factor is a random float between 0.8 and 1.2
            s = np.random.uniform(2-self.s_max, self.s_max) #float(np.random.choice([2-self.s_max, self.s_max]))

            tx, ty = np.random.uniform(-self.tx_max, self.tx_max), np.random.uniform(-self.ty_max, self.ty_max) 

            # Scale image by factor f, padding with zeros
            target = transforms.functional.affine(image, angle=r, translate=(tx, ty), 
                                                  scale=s, shear=sh)
            
            if self.bx_max != 0 or self.by_max != 0:
                # Squeeze the target image
                # Sample either plus or minus bx and by
                target = SCT(target.squeeze(), b=[bx,by])


            # Apply transforms to image and target
            image = self.transforms(image)
            target = self.transforms(target)

            #target = transforms.functional.rotate(image, angle_2)
            # Place image and target into arrays
            self.images[i] = image
            self.targets[i] = target

        self.images = self.images/1.0
        self.targets = self.targets/1.0
        batch_size = self.images.shape[0]

        # flat_images = self.images.view(batch_size, -1)
        # flat_targets = self.targets.view(batch_size, -1)   

        # # Solve system and subtract idnetity matrix
        # self.tG_est = None #TODO: Solve for tG_est
        # # Imshpw the tG matrix red blue colormap, zero is white:
        # plt.imshow(self.tG_est, cmap='RdBu', vmin=-1, vmax=1)
        # plt.colorbar()
        # plt.show()

        # unsqueeze to add a channel dimension
        self.images = self.images.unsqueeze(1)
        self.targets = self.targets.unsqueeze(1)



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target
    
# The following function takes the images in the dataset and performs
# the mathematical inversion defined by dividing the location vector
# by the norm squared of the location vector. This is done to the images
#
def inv(image):
    # Assumes the center of the image is the origin (0,0)
    # Get the dimensions of the image
    h, w = image.shape
    # Initialize an array to hold the inverted image
    inv_image = torch.zeros(h, w)
    # Loop through the pixels of the image and evualate the inverse
    for i in range(h):
        for j in range(w):
            # Get the pixel value
            x = image[i, j]
            # Get the location vector
            r = torch.tensor([i - h/2, j - w/2])
            # Get the norm squared of the location vector
            r_norm_sq = torch.sum(r**2)
            # Evaluate the inverse
            inv_image[i, j] = x/r_norm_sq
    return inv_image


class SyMNISTDataLoader(BaseDataLoader):
    """
    SyMNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, tf_range=(0, 0, 0, 1, 0)):
        
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        post_transform = transforms.Compose([
            # transforms.Normalize((0.1307,), (0.3081,))
            transforms.Normalize((0.47336,), (0.2393,))
        ])
        self.data_dir = data_dir
        self.dataset = SyMNIST(self.data_dir, tf_range, train=training, pre_transform=pre_transform, post_transform=post_transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class UnnormalSyMNISTDataLoader(BaseDataLoader):
    """
    SyMNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, tf_range=(0, 0, 0, 1, 0)):
        
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        post_transform = transforms.Compose([
            lambda x: x
        ])
        self.data_dir = data_dir
        self.dataset = SyMNIST(self.data_dir, tf_range, train=training, pre_transform=pre_transform, post_transform=post_transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SuperSyMNIST(Dataset):
    def __init__(self, data_dir, tf_range=(0, 0, 0, 1, 0, 0, 0), same_digit=True, dataset='mnist', pre_transform=None, post_transform=None, n=28, **kwargs):
        if dataset == 'mnist':    
            self.data = datasets.MNIST(data_dir, download=True, transform=pre_transform, **kwargs)
            self.n = n  # Canvas size (n x n)
        else:
            raise ValueError("Only 'mnist' dataset is supported.")

        self.tx_max, self.ty_max, self.r_max, self.s_max, self.sh_max, self.bx_max, self.by_max = tf_range
        self.same_digit = same_digit
        self.transforms = post_transform

        # Precompute label-based indices for faster pairing
        self.label_to_indices = {
            label: (self.data.targets == label).nonzero(as_tuple=True)[0].tolist()
            for label in range(10)
        }

        # Initialize arrays to hold precomputed images and targets
        self.images = torch.zeros(len(self.data), 1, self.n, self.n)
        self.targets = torch.zeros(len(self.data), 1, self.n, self.n)

        for i, (image, label) in enumerate(self.data):
            label = int(label)
            # Get a paired digit (target) based on the `same_digit` flag
            if self.same_digit:
                pair_idx = np.random.choice(self.label_to_indices[label])
            else:
                other_labels = [l for l in range(10) if l != label]
                random_label = np.random.choice(other_labels)
                pair_idx = np.random.choice(self.label_to_indices[random_label])

            target = self.data[pair_idx][0]

            # Apply non-translation transformations to the target
            r_mean = np.random.choice([-1, 0, 1])*self.r_max
            r = np.random.normal(r_mean, 0.0)

            tx = np.random.choice([-1, 1])*self.tx_max
            ty = np.random.uniform(-self.ty_max, self.ty_max)
            
            
            sh = np.random.uniform(-self.sh_max, self.sh_max)
            bx = np.random.uniform(-self.bx_max, self.bx_max)
            by = np.random.uniform(-self.by_max, self.by_max)
            s = np.random.uniform(2 - self.s_max, self.s_max)
            # tx, ty = np.random.uniform(-self.tx_max, self.tx_max), np.random.uniform(-self.ty_max, self.ty_max)
            

            target = transforms.functional.affine(target, angle=r, translate=(0, 0), scale=s, shear=sh)

            if self.bx_max != 0 or self.by_max != 0:
                target = SCT(target.squeeze(), b=[bx, by])

            # Create a larger canvas and place image and target at the center
            image_canvas = torch.zeros((1, self.n, self.n))  
            target_canvas = torch.zeros((1, self.n, self.n))

            img_size = image.shape[-1]
            max_offset = self.n - img_size
            center_offset = max_offset // 2

            # Place the original image at the center of the canvas
            image_canvas[0, center_offset:center_offset + img_size, center_offset:center_offset + img_size] = image

            # Place the transformed target at the center of the canvas
            target_canvas[0, center_offset:center_offset + img_size, center_offset:center_offset + img_size] = target

            # Apply translation transformation to the target canvas
            target_canvas = transforms.functional.affine(target_canvas, angle=0, translate=(tx, ty), scale=1, shear=0)

            # Apply post-transformations
            image_canvas = self.transforms(image_canvas) if self.transforms else image_canvas
            target_canvas = self.transforms(target_canvas) if self.transforms else target_canvas

            # Store precomputed images and targets
            self.images[i] = image_canvas
            self.targets[i] = target_canvas

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

class SuperSyMNISTDataLoader(BaseDataLoader):
    """
    SyMNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, tf_range=(0, 0, 0, 1, 0)):
        
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        post_transform = transforms.Compose([
            transforms.Normalize((0.47336,), (0.2393,))
        ])
        self.data_dir = data_dir
        self.dataset = SuperSyMNIST(self.data_dir, 
                                    tf_range, 
                                    train=training, 
                                    pre_transform=pre_transform, 
                                    post_transform=post_transform)


        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
