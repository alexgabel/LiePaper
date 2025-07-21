from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

import matplotlib.pyplot as plt
import os
import requests
from tqdm import tqdm


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
        elif dataset == 'galaxy':
            import h5py
            from PIL import Image
            from pathlib import Path
            from sklearn.model_selection import train_test_split

            galaxy_dir = Path(data_dir) / "Galaxy"
            galaxy_file = galaxy_dir / "Galaxy10_DECals.h5"

            if not galaxy_file.exists():
                print(f"Downloading Galaxy10_DECals.h5 into {galaxy_dir}")
                galaxy_dir.mkdir(parents=True, exist_ok=True)
                url = "https://zenodo.org/record/10845026/files/Galaxy10_DECals.h5"
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(galaxy_file, "wb") as file, tqdm(
                    desc="Galaxy10_DECals.h5",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        bar.update(len(data))
                print("Download complete.")
            else:
                print("Galaxy10 dataset already exists at:", galaxy_file)

            with h5py.File(galaxy_file, 'r') as f:
                images = np.array(f['images'])  # [17736, 256, 256, 3]
                labels = np.array(f['ans'], dtype=np.int64)

            # Convert to grayscale by averaging over RGB channels
            images = images.mean(axis=-1)

            # Perform stratified train/test split
            indices = np.arange(len(labels))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.1, random_state=42, stratify=labels
            )
            if kwargs.get("train", True):
                selected_images = images[train_idx]
            else:
                selected_images = images[test_idx]

            # Convert to tensor and normalize
            selected_images = torch.tensor(selected_images, dtype=torch.float32) / 255.0

            # Define transform pipeline (to be applied later)
            transform_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(56),
                transforms.Resize(28),
                transforms.ToTensor()
            ])

            # Store raw images, transform later in __init__ loop
            self.data = [(img, 0) for img in selected_images]
            self.n = 28
        else:
            raise ValueError("Only 'mnist' and 'galaxy' datasets are currently supported.")

            # Reduce the size of the dataset for testing purposes:
            # self.data = Subset(self.data, range(128))
        # n is the size of the image (H)
        
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

            # Apply non-translation transformations to the target
            # r_mean = np.random.choice([-2,-1, 0, 1,2])*self.r_max
            # r = np.random.normal(0, self.r_max)
            r = np.random.uniform(-self.r_max, self.r_max)

            tx = np.random.uniform(-self.tx_max, self.tx_max)
            ty = np.random.uniform(-self.ty_max, self.ty_max)
            
            sh = np.random.uniform(-self.sh_max, self.sh_max)
            bx = np.random.normal(0, self.bx_max)
            by = np.random.normal(0, self.by_max)
            # s = np.random.choice([2 - self.s_max,1.0, self.s_max])*1.0
            s = np.random.uniform(2 - self.s_max, self.s_max)
            # s = np.random.normal(1.0, self.s_max-1.0)
            # tx, ty = np.random.uniform(-self.tx_max, self.tx_max), np.random.uniform(-self.ty_max, self.ty_max)
            
            # Prepare the target image
            target = image.clone()

            # Apply geometric transformations to target
            if dataset == 'galaxy':
                target = transforms.functional.affine(target.unsqueeze(0), angle=r, translate=(tx, ty),
                                                      scale=s, shear=sh).squeeze(0)
            else:
                target = transforms.functional.affine(target, angle=r, translate=(tx, ty),
                                                      scale=s, shear=sh)
            
            if self.bx_max != 0 or self.by_max != 0:
                # Squeeze the target image
                # Sample either plus or minus bx and by
                target = SCT(target.squeeze(), b=[bx,by])

            # For galaxy dataset, apply transform_pipeline after all geometric transforms
            if dataset == 'galaxy':
                image = transform_pipeline(image)
                target = transform_pipeline(target)

            # Apply post transforms to image and target
            image = self.transforms(image)
            target = self.transforms(target)

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

        # Visualize first few image pairs to debug transformations
        num_vis = min(5, len(self.images))
        fig, axs = plt.subplots(num_vis, 2, figsize=(6, num_vis * 2))
        for i in range(num_vis):
            axs[i, 0].imshow(self.images[i, 0].numpy(), cmap='gray')
            axs[i, 0].set_title('Original')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(self.targets[i, 0].numpy(), cmap='gray')
            axs[i, 1].set_title('Transformed')
            axs[i, 1].axis('off')
        plt.tight_layout()
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/sample_visualization.png")
        plt.close()



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        return image, target
    

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
        self.dataset = SyMNIST(self.data_dir, tf_range, dataset='mnist', train=training, pre_transform=pre_transform, post_transform=post_transform)
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
        # Import required modules for galaxy dataset
        import_types = False
        if dataset == 'galaxy':
            import_types = True
        if import_types:
            import h5py
            from PIL import Image
            from pathlib import Path
            from sklearn.model_selection import train_test_split

        if dataset == 'mnist':
            self.data = datasets.MNIST(data_dir, download=True, transform=pre_transform, **kwargs)
            self.n = n  # Canvas size (n x n)
            # Precompute label-based indices for faster pairing
            self.label_to_indices = {
                label: (self.data.targets == label).nonzero(as_tuple=True)[0].tolist()
                for label in range(10)
            }
            self.transform_pipeline = None
        elif dataset == 'galaxy':
            # Galaxy10_DECals dataset loading and preparation
            from pathlib import Path
            import h5py
            from PIL import Image
            from sklearn.model_selection import train_test_split

            galaxy_dir = Path(data_dir) / "Galaxy"
            galaxy_file = galaxy_dir / "Galaxy10_DECals.h5"

            if not galaxy_file.exists():
                print(f"Downloading Galaxy10_DECals.h5 into {galaxy_dir}")
                galaxy_dir.mkdir(parents=True, exist_ok=True)
                url = "https://zenodo.org/record/10845026/files/Galaxy10_DECals.h5"
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                with open(galaxy_file, "wb") as file, tqdm(
                    desc="Galaxy10_DECals.h5",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
                        bar.update(len(data))
                print("Download complete.")
            else:
                print("Galaxy10 dataset already exists at:", galaxy_file)

            with h5py.File(galaxy_file, 'r') as f:
                images = np.array(f['images'])  # [17736, 256, 256, 3]
                labels = np.array(f['ans'], dtype=np.int64)

            # Convert to grayscale by averaging over RGB channels
            images = images.mean(axis=-1)  # shape: [N, 256, 256]

            # Stratified train/test split
            indices = np.arange(len(labels))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.1, random_state=42, stratify=labels
            )
            if kwargs.get("train", True):
                selected_images = images[train_idx]
                selected_labels = labels[train_idx]
            else:
                selected_images = images[test_idx]
                selected_labels = labels[test_idx]

            # Set the image size to 28x28
            self.n = 28
            # Define transform pipeline to resize to 28x28
            transform_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(56),
                transforms.Resize(self.n),
                transforms.ToTensor()
            ])
            self.transform_pipeline = transform_pipeline

            # Convert images to tensors and normalize to [0,1]
            # Store as list of (image_tensor, label)
            self.data = []
            for img, lbl in zip(selected_images, selected_labels):
                # img: [256,256] float
                img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0  # [256,256], float32, [0,1]
                # Store as (img_tensor, label); transform_pipeline will be applied later
                self.data.append((img_tensor, int(lbl)))

            # Build label_to_indices dict
            unique_labels = np.unique(selected_labels)
            self.label_to_indices = {
                int(label): [i for i, (img, lbl) in enumerate(self.data) if lbl == int(label)]
                for label in unique_labels
            }
        else:
            raise ValueError("Only 'mnist' and 'galaxy' datasets are supported.")

        self.tx_max, self.ty_max, self.r_max, self.s_max, self.sh_max, self.bx_max, self.by_max = tf_range
        self.same_digit = same_digit
        self.transforms = post_transform

        # Initialize arrays to hold precomputed images and targets
        self.images = torch.zeros(len(self.data), 1, self.n, self.n)
        self.targets = torch.zeros(len(self.data), 1, self.n, self.n)

        for i, (image, label) in enumerate(self.data):
            label = int(label)
            # Get a paired digit (target) based on the `same_digit` flag
            if self.same_digit:
                pair_idx = np.random.choice(self.label_to_indices[label])
            else:
                # Choose a random label different from current label
                other_labels = [l for l in self.label_to_indices if l != label]
                random_label = np.random.choice(other_labels)
                pair_idx = np.random.choice(self.label_to_indices[random_label])

            target = self.data[pair_idx][0]

            # Apply geometric transformation to target (rotation, scaling, shear)
            r = np.random.uniform(-self.r_max, self.r_max)
            tx = np.random.uniform(-self.tx_max, self.tx_max)
            ty = np.random.uniform(-self.ty_max, self.ty_max)
            sh = np.random.uniform(-self.sh_max, self.sh_max)
            bx = np.random.uniform(-self.bx_max, self.bx_max)
            by = np.random.uniform(-self.by_max, self.by_max)
            s = np.random.uniform(2 - self.s_max, self.s_max)

            target_trans = transforms.functional.affine(target.unsqueeze(0), angle=r, translate=(0, 0), scale=s, shear=sh).squeeze(0)

            # Apply SCT if needed
            if self.bx_max != 0 or self.by_max != 0:
                target_trans = SCT(target_trans.squeeze(), b=[bx, by])

            # Translate AFTER transformation (now target is still 256x256)
            target_trans = transforms.functional.affine(target_trans.unsqueeze(0), angle=0, translate=(tx, ty), scale=1, shear=0).squeeze(0)

            # Resize both original and transformed image to self.n using pipeline
            if self.transform_pipeline:
                image = self.transform_pipeline(image)
                target_trans = self.transform_pipeline(target_trans)

            # Apply post-transforms (e.g., normalization)
            image = self.transforms(image) if self.transforms else image
            target_trans = self.transforms(target_trans) if self.transforms else target_trans

            # Store precomputed images and targets
            self.images[i] = image
            self.targets[i] = target_trans

        # Visualize first few image pairs to debug transformations
        num_vis = min(5, len(self.images))
        fig, axs = plt.subplots(num_vis, 2, figsize=(6, num_vis * 2))
        for i in range(num_vis):
            axs[i, 0].imshow(self.images[i, 0].numpy(), cmap='gray')
            axs[i, 0].set_title('Original')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(self.targets[i, 0].numpy(), cmap='gray')
            axs[i, 1].set_title('Transformed')
            axs[i, 1].axis('off')
        plt.tight_layout()
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/sample_visualization.png")
        plt.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

    def visualize_samples(self, num_samples=5):
        """
        Visualize a few sample pairs (image and target) with purple-yellow colormap.
        """
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 2))
        for i in range(num_samples):
            image, target = self.images[i, 0].numpy(), self.targets[i, 0].numpy()
            axes[i, 0].imshow(image, cmap='viridis')  # Purple-yellow colormap
            axes[i, 0].axis('off')
            axes[i, 0].set_title("Original Image")

            axes[i, 1].imshow(target, cmap='viridis')  # Purple-yellow colormap
            axes[i, 1].axis('off')
            axes[i, 1].set_title("Transformed Target")

        plt.tight_layout()
        plt.show()

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


class GalaxSymDataLoader(BaseDataLoader):
    """
    GalaxSym data loader for transformed Galaxy10 dataset with symmetry-aware augmentations.
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, tf_range=(0, 0, 0, 1, 0, 0, 0)):
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        post_transform = transforms.Compose([
            transforms.Normalize((0.4400,), (0.2344,))
        ])
        self.data_dir = data_dir
        self.dataset = SyMNIST(self.data_dir, 
                               tf_range=tf_range, 
                               dataset='galaxy', 
                               train=training, 
                               pre_transform=pre_transform, 
                               post_transform=post_transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SuperGalaxSymDataLoader(BaseDataLoader):
    """
    SuperGalaxSym data loader for paired Galaxy10 images with symmetry-aware augmentation.
    One image is transformed, both belong to the same class.
    """
    def __init__(self, data_dir, batch_size, 
                 shuffle=True, validation_split=0.0, 
                 num_workers=1, training=True, tf_range=(0, 0, 0, 1, 0, 0, 0)):
        
        pre_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        post_transform = transforms.Compose([
            transforms.Normalize((0.4400,), (0.2344,))
        ])

        self.data_dir = data_dir
        self.dataset = SuperSyMNIST(self.data_dir, 
                                    tf_range=tf_range, 
                                    same_digit=True,
                                    dataset='galaxy',
                                    train=training, 
                                    pre_transform=pre_transform, 
                                    post_transform=post_transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)