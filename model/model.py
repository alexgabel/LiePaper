import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from base import BaseModel
from scipy.linalg import expm

class tNet(BaseModel):
    '''Simple MLP that takes input and target images and outputs rotation angle'''
    def __init__(self, input_size, hidden_sizes, tnum=1):
        super().__init__()
        self.tnum = tnum # originally 1
        layer_sizes = [input_size] + hidden_sizes + [self.tnum]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(BaseModel):
    def __init__(self, input_size, hidden_sizes, latent_dim, layer_norm=False):
        super().__init__()
        layer_sizes = [input_size] + hidden_sizes + [latent_dim]
        layers = []

        if not layer_norm:
            for i in range(len(layer_sizes)-1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < len(layer_sizes)-2:
                    layers.append(nn.LeakyReLU(0.2))
            self.layers = nn.Sequential(*layers)
        else:
            for i in range(len(layer_sizes)-1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < len(layer_sizes)-2:
                    layers.append(nn.LayerNorm(layer_sizes[i+1]))
                    layers.append(nn.LeakyReLU(0.2))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(BaseModel):
    def __init__(self, latent_dim, hidden_sizes, output_size, layer_norm=False):
        super().__init__()
        # Flip hidden sizes for decoder
        hidden_sizes = hidden_sizes[::-1]
        layer_sizes = [latent_dim] + hidden_sizes + [output_size]
        layers = []

        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                if layer_norm:
                    layers.append(nn.LayerNorm(layer_sizes[i+1]))
                layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Decoder with dropout layers:
class DecoderDropout(BaseModel):
    def __init__(self, latent_dim, hidden_sizes, output_size, dropout=0.5):
        super().__init__()
        # Flip hidden sizes for decoder
        hidden_sizes = hidden_sizes[::-1]
        layer_sizes = [latent_dim] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EncoderLieTDecoder(BaseModel):
    
    def __init__(self, input_size, hidden_sizes, t_hidden_sizes, latent_dim, non_affine=False, channels=1, dropout=False, conv=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.c = channels
        self.conv = conv

        self.encoder = Encoder(input_size, hidden_sizes, channels*latent_dim, layer_norm=False)

        # Initialize 6 trainable parameters for basis coefficients:
        if non_affine:
            self.a = nn.Parameter(torch.randn(12), requires_grad=True)
        else:    
            self.a = nn.Parameter(torch.randn(6), requires_grad=True)
            # Make a [0,0,-1,0,1,0] vector:
            # self.a.data = torch.tensor([1,1,-1,0,1,0]).float().to("cuda:0")
        print("a_init = ", self.a / torch.norm(self.a, p=2))

        self.D = self.generate_basis(d_squared=self.latent_dim, non_affine=non_affine).clone().detach().to("cuda:0")
        self.D_big = self.generate_basis(d_squared=input_size, non_affine=non_affine).clone().detach().to("cuda:0")

        if dropout is True:
            self.decoder = DecoderDropout(channels*latent_dim, hidden_sizes, input_size, dropout=0.5)
        else:
            self.decoder = Decoder(channels*latent_dim, hidden_sizes, input_size, layer_norm=False)

        self.t = tNet(2*input_size, t_hidden_sizes)

        self.i = 0

    def forward(self, x, epsilon=0, tMax=0, zTest=False):
        # Normalize a vector by dividing by its norm:
        a_norm = torch.norm(self.a, p=2)
        a_normed = self.a / a_norm

        self.G = torch.einsum('i, imn -> mn', a_normed, self.D)

        x, y = torch.split(x, 1, dim=1)

        # Make a copy of x with correct shape for the cnn
        # Save size before flattening input:
        data_shape = x.size()

        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
  
        
        # For zero epsilon, use the tNet, else set t to epsilon:
        if epsilon == 0 and tMax == 0:
            # Get t from concatenating x and y along the feature dimension
            # and passing through tNet:
            t = self.t(torch.cat((x,y), dim=1))
            t = t.squeeze()

        elif epsilon != 0 and tMax == 0 and self.i < 1:
            t = epsilon * torch.ones(x.size(0)).to("cuda:0")
        elif tMax > 0 and epsilon == 0 and self.i < 1:
            # tMax sets the maximum value of t in the interval [-tMax, tMax]
            # in which the number of steps agrees with the batch size:
            t = torch.linspace(-tMax, tMax, x.size(0)).to("cuda:0")


        z = self.encoder(x)
        # Reshape output into self.c channels:
        z = z.view(-1, self.c, self.latent_dim)

        exponent = torch.einsum('i, mn -> imn', t, self.G)
        exptG = torch.matrix_exp(exponent)

        z_t = torch.einsum('ica, iba -> icb', z, exptG)
        # Reshape output into self.c*latent_dim channels:
        z_t_flat = z_t.view(-1, self.c*self.latent_dim)
        x_t = self.decoder(z_t_flat)
        
        ### PLOTTING FOR Z-TESTING ###
        # The following code is the z-test. It plots the latent vector z 
        # and the transformed latent vector z_t as latent_dim by latent_dim 
        # images. It also plots the flow of the generator G by plotting its
        # action on a two-dimensional gaussian blob. Also, arrows are plotted
        # that correspond to the flow of the generator G (by using the vector
        # field form of the generator G) on the gaussian blob. 

        if zTest and self.i < 1:
            ground_truth_a = torch.tensor([0, 0, -1,0,0,0,0, 1, 0,0,0,0], device=self.a.device, dtype=torch.float32)
        
            # Calculate G loss
            g_loss = self.calculate_g_loss(ground_truth_a)

            # Print G Loss components
            print("G Loss Components:")
            print(f"  Total MSE: {g_loss['g_mse']:.4f} (Neg: {g_loss['neg_g_mse']:.4f})")
            print(f"  Drift MSE: {g_loss['g_drift_mse']:.4f} (Neg: {g_loss['neg_g_drift_mse']:.4f})")
            print(f"  Diffusion MSE: {g_loss['g_diff_mse']:.4f} (Neg: {g_loss['neg_g_diff_mse']:.4f})")
            if self.a.size(0) == 12:  # Non-affine terms
                print(f"  Non-Affine MSE: {g_loss['g_nonaffine_mse']:.4f} (Neg: {g_loss['neg_g_nonaffine_mse']:.4f})")

            # Visualize the generator field and eigenvectors
            self.visualize(z, z_t, t, a_normed, ground_truth_a, plot_eigen=False)
            self.plot_patches(z, z_t, t, ground_truth_a)
            self.i += 1
        ################################    
        x_t = x_t.view(data_shape)

        return x_t, exptG, t

    def normal(self, x):
        # Save size before flattening input:
        data_shape = x.size()
        
        if not self.conv:
            # Put x in correct shape:
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x = self.decoder(z)

            # Unflattens output so it has same shape as input:
            x = x.view(data_shape)
        else:
            z = self.encoder(x)
            x = self.decoder(z)
            x = x.view(data_shape)

        return x
    
    def matrix_plot(self,Lx):
        """
        Plots the matrix Lx and its exponential side by side with colorbars.
        
        Args:
            Lx (numpy.ndarray or torch.Tensor): The matrix to be visualized.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Lx
        im1 = axes[0].imshow(Lx, cmap="viridis", aspect="auto")
        axes[0].set_title("Lx (Derivative Matrix)")
        axes[0].set_xlabel("Columns")
        axes[0].set_ylabel("Rows")
        fig.colorbar(im1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)

        # Plot expm(Lx)
        im2 = axes[1].imshow(expm(Lx), cmap="viridis", aspect="auto")
        axes[1].set_title("expm(Lx) (Exponential of Lx)")
        axes[1].set_xlabel("Columns")
        axes[1].set_ylabel("Rows")
        fig.colorbar(im2, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)

        # Add some space between plots
        plt.tight_layout()

        # Show the plots
        plt.show()

    def generate_basis(self, d_squared, non_affine=False, basis_type='sw'):
        # L0 = lambda d,z: np.sum([2*np.pi*p/d**2 * np.sin(2*np.pi*p/d *z) 
        #                          for p in np.arange(-d/2+1,d/2)], axis=0)
        # Latent dimension:
        d = int(np.sqrt(d_squared))

        coords = np.mgrid[:d,:d] - d/2
        x,y = coords.reshape((2,-1))

        # Swap coordinates and make y -> -y to match image coordinates:
        x,y = y,x
        y = -y

        dx = (x[:, np.newaxis] - x) * (y[:, np.newaxis] == y)
        dy = (y[:, np.newaxis] - y) * (x[:, np.newaxis] == x)

        if basis_type == "sw":
            # Shannon-Whittaker interpolation
            L0 = lambda d, z: np.sum(
                [2 * np.pi * p / d**2 * np.sin(2 * np.pi * p / d * z) for p in np.arange(-d / 2 + 1, d / 2)], 
                axis=0
            )
        elif basis_type == "exp":
            L0 = lambda d, z: np.sum(
                [np.exp(-np.abs(p / d)) * 2 * np.pi * p / d**2 * np.sin(2 * np.pi * p / d * z) for p in np.arange(-d / 2 + 1, d / 2)], 
                axis=0
            )
        elif basis_type == "gauss":
            L0 = lambda d, z: np.sum(
                [np.exp(-p**2 / (2 * (d / 4)**2)) * 2 * np.pi * p / d**2 * np.sin(2 * np.pi * p / d * z) for p in np.arange(-d / 2 + 1, d / 2)], 
                axis=0
            )
        else:
            raise ValueError(f"Invalid basis_type: {basis_type}. Choose 'shannon', 'spline', or 'gaussian'.")

        Lx = L0(2*d, dx)
        Ly = L0(2*d, dy)
    
        xLx = np.diag(x) @ Lx
        yLx = np.diag(y) @ Lx
        xLy = np.diag(x) @ Ly
        yLy = np.diag(y) @ Ly

        if non_affine:
            # Do the same for x_squared and y_squared, and x*y:
            x_squared = np.diag(x)**2
            y_squared = np.diag(y)**2
            xy = np.diag(x) @ np.diag(y)
            
            x2Lx = x_squared @ Lx
            y2Lx = y_squared @ Lx
            xyLx = xy @ Lx
            x2Ly = x_squared @ Ly
            y2Ly = y_squared @ Ly
            xyLy = xy @ Ly

            D = np.stack([Lx, xLx, yLx, x2Lx,  xyLx, y2Lx, Ly, xLy, yLy, x2Ly,  xyLy, y2Ly], axis=0)
            D = torch.from_numpy(D).float()
        else:
            D = np.stack([Lx, xLx, yLx, Ly, xLy, yLy], axis=0)
            D = torch.from_numpy(D).float()

        return D
    
    def calculate_g_loss(self, ground_truth_a):
        """
        Calculate MSE losses for the learned `G` compared to the ground truth.

        Handles both affine (`a` length 6) and non-affine (`a` length 12) cases.
        """
        # Normalize ground truth coefficients
        ground_truth_a = torch.tensor(ground_truth_a, device=self.a.device, dtype=torch.float32)
        gt_norm = torch.norm(ground_truth_a, p=2)
        ground_truth_a = ground_truth_a / gt_norm
        a_normed = self.a / torch.norm(self.a, p=2)
        # Total MSE
        g_mse = torch.mean((a_normed - ground_truth_a) ** 2).item()
        neg_g_mse = torch.mean((a_normed + ground_truth_a) ** 2).item()

        if a_normed.size(0) == 6:  # Affine case
            # Drift components
            learned_drift = a_normed[[0, 3]]
            gt_drift = ground_truth_a[[0, 3]]

            # Diffusion components
            learned_diffusion = a_normed[[1, 2, 4, 5]]
            gt_diffusion = ground_truth_a[[1, 2, 4, 5]]

            # Non-affine components do not exist
            g_nonaffine_mse = neg_g_nonaffine_mse = 0.0

        elif a_normed.size(0) == 12:  # Non-affine case
            # Drift components
            learned_drift = a_normed[[0, 6]]
            gt_drift = ground_truth_a[[0, 6]]

            # Diffusion components
            learned_diffusion = a_normed[[1, 2, 7, 8]]
            gt_diffusion = ground_truth_a[[1, 2, 7, 8]]

            # Non-affine components
            learned_nonaffine = a_normed[[3, 4, 5, 9, 10, 11]]
            gt_nonaffine = ground_truth_a[[3, 4, 5, 9, 10, 11]]
            g_nonaffine_mse = torch.mean((learned_nonaffine - gt_nonaffine) ** 2).item()
            neg_g_nonaffine_mse = torch.mean((learned_nonaffine + gt_nonaffine) ** 2).item()

        else:
            raise ValueError("Unexpected size for `a` vector. Supported lengths are 6 (affine) or 12 (non-affine).")

        # Drift MSE
        g_drift_mse = torch.mean((learned_drift - gt_drift) ** 2).item()
        neg_g_drift_mse = torch.mean((learned_drift + gt_drift) ** 2).item()

        # Diffusion MSE
        g_diff_mse = torch.mean((learned_diffusion - gt_diffusion) ** 2).item()
        neg_g_diff_mse = torch.mean((learned_diffusion + gt_diffusion) ** 2).item()

        return {
            'g_mse': g_mse,
            'neg_g_mse': neg_g_mse,
            'g_drift_mse': g_drift_mse,
            'neg_g_drift_mse': neg_g_drift_mse,
            'g_diff_mse': g_diff_mse,
            'neg_g_diff_mse': neg_g_diff_mse,
            'g_nonaffine_mse': g_nonaffine_mse,
            'neg_g_nonaffine_mse': neg_g_nonaffine_mse
        }

    def visualize(self, z, z_t, t, a_normed, ground_truth_a, plot_eigen=False):
        # Generator visualization
        grid_size = 10
        # Get d from square root of latent dimension
        d = int(np.sqrt(self.latent_dim))
        x_vals = np.linspace(-d, d, grid_size)
        y_vals = np.linspace(-d, d, grid_size)
        X, Y = np.meshgrid(x_vals, y_vals)
        X, Y = Y, X
        Y = -Y

        if self.D.shape[0] == 6:  # Affine case
            a = a_normed[:6].detach().cpu().numpy()
            U = a[0] + a[1] * X + a[2] * Y
            V = a[3] + a[4] * X + a[5] * Y
            drift = np.array([a[0], a[3]])
            diffusion = np.array([[a[1], a[2]], [a[4], a[5]]])
        elif self.D.shape[0] == 12:  # Non-affine case
            a = a_normed[:12].detach().cpu().numpy()
            U = a[0] + a[1] * X + a[2] * Y + a[3] * X**2 + a[4] * X * Y + a[5] * Y**2
            V = a[6] + a[7] * X + a[8] * Y + a[9] * X**2 + a[10] * X * Y + a[11] * Y**2
            drift = np.array([a[0], a[6]])
            diffusion = np.array([[a[1], a[2]], [a[7], a[8]]])

        # Eigenvalue and eigenvector calculation
        eigenvalues, eigenvectors = np.linalg.eig(diffusion)

        plt.figure(figsize=(8, 8))
        plt.quiver(X, Y, U, V, color='r', label='Generator Field')

        if plot_eigen:
            for idx, eigenvalue in enumerate(eigenvalues):
                if np.iscomplex(eigenvalue):  # Complex eigenvalue
                    plt.scatter(*drift, color='blue', s=100, label='Complex Eigenvalue')
                else:  # Real eigenvalue
                    eigenvector = eigenvectors[:, idx].real
                    plt.quiver(
                        drift[0], drift[1],
                        eigenvalue * eigenvector[0], eigenvalue * eigenvector[1],
                        angles='xy', scale_units='xy', scale=1,
                        color='blue', label=f"Eigenvector {idx+1}"
                    )

        # Calculate G loss
        g_loss = self.calculate_g_loss(ground_truth_a)

        # Determine argmin and select corresponding metrics
        argmin_g = 0 if g_loss['g_mse'] < g_loss['neg_g_mse'] else 1
        min_g_mse = [g_loss['g_mse'], g_loss['neg_g_mse']][argmin_g]
        min_drift_mse = [g_loss['g_drift_mse'], g_loss['neg_g_drift_mse']][argmin_g]
        min_diffusion_mse = [g_loss['g_diff_mse'], g_loss['neg_g_diff_mse']][argmin_g]

        # Add MSE information to the title
        title_info = (
            f"MSE: {min_g_mse:.4f}, Drift: {min_drift_mse:.4f}, Diff.: {min_diffusion_mse:.4f}"
        )
        plt.title(f"{title_info}")

        # plt.title("Generator Vector Field with Eigenvectors")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        # plt.legend(loc='upper left', fontsize=8)
        plt.show()

    def taylor_loss(self, combined_data, order=1):
        # Normalize a vector by dividing by its norm:
        a_norm = torch.norm(self.a, p=2)
        a_normed = self.a / a_norm

        self.G_big = torch.einsum('i, imn -> mn', a_normed, self.D_big)

        x, y = torch.split(combined_data, 1, dim=1)

        # Save size before flattening input:
        data_shape = x.size()

        # Flatten x and y:
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)   

        # Get t from concatenating x and y along the feature dimension
        # and passing through tNet:
        t = self.t(torch.cat((x,y), dim=1))
        # Flatten t:
        t = t.squeeze()
        # t does not require gradient:
        t_no_grad = t.detach()

        tG = torch.einsum('i, mn -> imn', t_no_grad, self.G_big)

        if order == 1:    
            # First order Taylor approximation:
            exptG_order = torch.eye(self.D_big.shape[1]).to("cuda:0") + tG
        elif order == 2:
            # Second order Taylor approximation:
            exptG_order = torch.eye(self.D_big.shape[1]).to("cuda:0") + tG + 0.5*torch.einsum('imn, ink -> imk', tG, tG)
        else:
            raise ValueError("Order must be 1 or 2.")

        x_t = torch.einsum('ia, iba -> ib', x, exptG_order)

        return x_t, y
    
    def plot_patches(self, z, z_t, t, ground_truth_a=None):
        latent_dim_sqrt = int(np.sqrt(self.latent_dim))
        print(z.shape, z_t.shape, t.shape)

        # Initialize the figure with 6 rows and 11 columns
        fig, axs = plt.subplots(6, 11, figsize=(20, 12))  # Adjusted height for better paper fit

        # Loop over the first 11 values to display z and z_t patches
        for j in range(11):
            # Plot the z patch in the first row
            z_plot = z[j, 0].view(latent_dim_sqrt, latent_dim_sqrt)
            axs[0, j].imshow(z_plot.detach().cpu().numpy(), cmap='viridis')
            axs[0, j].axis('off')
            axs[0, j].set_title(f't = {t[j].item():.2f}', fontsize=8)

            # Plot the z_t patch in the second row
            z_t_plot = z_t[j, 0].view(latent_dim_sqrt, latent_dim_sqrt)
            axs[1, j].imshow(z_t_plot.detach().cpu().numpy(), cmap='viridis')
            axs[1, j].axis('off')

        # Generate quasi-equally spaced t values between -1 and 1, including 0
        t_values = torch.linspace(-1, 1, 11, device='cuda:0')

        # Create sample patches
        z_temp1 = self.create_gaussian_line_segment(stddev=0.2)
        z_temp2 = self.create_gaussian_blob(stddev=0.5)

        # Define matrices G_test and G_model
        a_test = ground_truth_a.float().to("cuda:0")#torch.tensor([0,0,0,1, 0, -1,0,0,0, 0, 2, 0]).float().to("cuda:0")
        G_test = torch.einsum('i, imn -> mn', a_test / torch.norm(a_test, p=2), self.D)
        G_model = torch.einsum('i, imn -> mn', self.a / torch.norm(self.a, p=2), self.D)

        # Loop to plot transformations for line and blob
        for j in range(11):
            t_sample = t_values[j]

            # Apply matrix exponentiation and transformations for line
            exptG = torch.matrix_exp(torch.einsum('i,mn -> imn', t_sample.view(1), G_test))
            exptG_model = torch.matrix_exp(torch.einsum('i,mn -> imn', t_sample.view(1), G_model))
            z_t_temp = torch.einsum('ia, iba -> ib', z_temp1, exptG)
            z_t_temp_model = torch.einsum('ia, iba -> ib', z_temp1, exptG_model)

            # Plot in rows 3 and 4
            axs[2, j].imshow(z_t_temp_model.view(latent_dim_sqrt, latent_dim_sqrt).detach().cpu().numpy(), cmap='viridis')
            axs[2, j].axis('off')
            axs[3, j].imshow(z_t_temp.view(latent_dim_sqrt, latent_dim_sqrt).detach().cpu().numpy(), cmap='viridis')
            axs[3, j].axis('off')

            # Apply matrix exponentiation and transformations for blob
            z_t_temp = torch.einsum('ia, iba -> ib', z_temp2, exptG)
            z_t_temp_model = torch.einsum('ia, iba -> ib', z_temp2, exptG_model)

            # Plot in rows 5 and 6
            axs[4, j].imshow(z_t_temp_model.view(latent_dim_sqrt, latent_dim_sqrt).detach().cpu().numpy(), cmap='viridis')
            axs[4, j].axis('off')
            axs[5, j].imshow(z_t_temp.view(latent_dim_sqrt, latent_dim_sqrt).detach().cpu().numpy(), cmap='viridis')
            axs[5, j].axis('off')

        # Add vertical labels centered on rows and closer to the grid
        fig.text(0.02, 0.88, r'$\mathbf{z}_0$', va='center', ha='center', rotation='vertical', fontsize=12)
        fig.text(0.02, 0.72, r'$\mathbf{z}_t$', va='center', ha='center', rotation='vertical', fontsize=12)

        fig.text(0.02, 0.56, 'learnt TF\n(line)', va='center', ha='center', rotation='vertical', fontsize=12)
        fig.text(0.02, 0.40, 'GT TF\n(line)', va='center', ha='center', rotation='vertical', fontsize=12)

        fig.text(0.02, 0.24, 'learnt TF\n(blob)', va='center', ha='center', rotation='vertical', fontsize=12)
        fig.text(0.02, 0.08, 'GT TF\n(blob)', va='center', ha='center', rotation='vertical', fontsize=12)

        # Adjust layout and display
        plt.tight_layout(rect=[0.05, 0, 1, 1])  # Adjust rect to bring labels closer
        plt.show()






    def create_gaussian_blob(self, stddev=0.5):
        # Create a 2D grid of coordinates
        latent_dim_sqrt = int(np.sqrt(self.latent_dim))
        x = torch.linspace(-1, 1, latent_dim_sqrt)
        y = torch.linspace(-1, 1, latent_dim_sqrt)
        X, Y = torch.meshgrid(x, y)

        # Calculate the 2D Gaussian function
        gaussian_blob = torch.exp(-(((X)**2 + (Y)**2) / (2 * stddev**2)))
        # Rotate the Gaussian blob by 90 degrees

        gaussian_blob= torch.rot90(gaussian_blob,k=1)

        # Flatten the Gaussian blob to match the latent vector dimensions
        gaussian_blob = gaussian_blob.reshape(1, self.latent_dim).to("cuda:0")
        return gaussian_blob
    
    def create_gaussian_line_segment(self, stddev=0.5):
        # Create a 2D grid of coordinates
        latent_dim_sqrt = int(np.sqrt(self.latent_dim))
        x = torch.linspace(-1, 1, latent_dim_sqrt)
        y = torch.linspace(-1, 1, latent_dim_sqrt)
        X, Y = torch.meshgrid(x, y)

        # Calculate the 2D Gaussian line
        gaussian_blob = torch.exp(-((X**2 ) / (2 * stddev**2)))
        
        # Set the first and last row to zero
        gaussian_blob[:, 0] = 0
        gaussian_blob[:, -1] = 0

        # Flatten the Gaussian blob to match the latent vector dimensions
        gaussian_blob = gaussian_blob.reshape(1, self.latent_dim).to("cuda:0")
        return gaussian_blob
    
    def create_L_shape(length=5, width=1):
        """
        Create an L-shaped figure.
        
        Parameters:
        - length: Length of the arms of the L shape.
        - width: Width of the arms of the L shape.
        
        Returns:
        A tensor representing the L-shaped figure.
        """
        L_shape = torch.zeros((1, length, length)).to("cuda:0")  # Create a blank square tensor
        L_shape[0, :length, :width] = 1  # Vertical arm
        L_shape[0, :width, :length] = 1  # Horizontal arm
        return L_shape


class EncoderLieMulTDecoder(BaseModel):
    
    def __init__(self, input_size, hidden_sizes, t_hidden_sizes, latent_dim, non_affine=False, channels=4, dropout=False, conv=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.c = channels
        self.conv = conv

        self.encoder = Encoder(input_size, hidden_sizes, channels*latent_dim, layer_norm=False)

        # Initialize 6 trainable parameters for basis coefficients:
        if non_affine:
            self.a = nn.Parameter(torch.randn(12), requires_grad=True)
            self.a2 = nn.Parameter(torch.randn(12), requires_grad=True)
        else:    
            self.a = nn.Parameter(torch.randn(6), requires_grad=True)
            self.a2 = nn.Parameter(torch.randn(6), requires_grad=True)
            # Make a [0,0,-1,0,1,0] vector:
            # self.a.data = torch.tensor([1,1,-1,0,1,0]).float().to("cuda:0")
        print("a_init = ", self.a / torch.norm(self.a, p=2))
        print("a2_init = ", self.a2 / torch.norm(self.a2, p=2))

        self.D = self.generate_basis(d_squared=self.latent_dim, non_affine=non_affine).clone().detach().to("cuda:0")
        self.D_big = self.generate_basis(d_squared=input_size, non_affine=non_affine).clone().detach().to("cuda:0")

        if dropout is True:
            self.decoder = DecoderDropout(channels*latent_dim, hidden_sizes, input_size, dropout=0.5)
        else:
            self.decoder = Decoder(channels*latent_dim, hidden_sizes, input_size, layer_norm=False)

        self.t = tNet(2*input_size, t_hidden_sizes, tnum=3)

        self.i = 0

    def forward(self, x, epsilon=0, tMax=0, zTest=False):
        # Normalize a vector by dividing by its norm:
        a_norm = torch.norm(self.a, p=2)
        a_normed = self.a / a_norm

        a2_norm = torch.norm(self.a2, p=2)
        a2_normed = self.a2 / a2_norm

        self.G = torch.einsum('i, imn -> mn', a_normed, self.D)
        self.G2 = torch.einsum('i, imn -> mn', a2_normed, self.D)
        
        commutator = torch.matmul(self.G, self.G2) - torch.matmul(self.G2, self.G)

        x, y = torch.split(x, 1, dim=1)

        # Make a copy of x with correct shape for the cnn
        # Save size before flattening input:
        data_shape = x.size()

        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
  
        
        # For zero epsilon, use the tNet, else set t to epsilon:
        if epsilon == 0 and tMax == 0:
            # Get t from concatenating x and y along the feature dimension
            # and passing through tNet:
            out_t= self.t(torch.cat((x,y), dim=1))
            t, t2, t12 = out_t[:,0], out_t[:,1], out_t[:,2]

        elif epsilon != 0 and tMax == 0 and self.i < 1:
            t = epsilon * torch.ones(x.size(0)).to("cuda:0")
        elif tMax > 0 and epsilon == 0 and self.i < 1:
            # tMax sets the maximum value of t in the interval [-tMax, tMax]
            # in which the number of steps agrees with the batch size:
            t = torch.linspace(-tMax, tMax, x.size(0)).to("cuda:0")




        exponent = torch.einsum('i, mn -> imn', t, self.G)
        exponent2 = torch.einsum('i, mn -> imn', t2, self.G2)
        exponent12 = torch.einsum('i, mn -> imn', t12, commutator)
        exptG = torch.matrix_exp(exponent)
        exptG2 = torch.matrix_exp(exponent2)
        exptG12 = torch.matrix_exp(exponent12)

        z = self.encoder(x)
        # Reshape output into self.c channels:
        z = z.view(-1, self.c, self.latent_dim)

        z_t = torch.einsum('ica, iba -> icb', z, exptG)
        z_t = torch.einsum('ica, iba -> icb', z_t, exptG2)
        z_t = torch.einsum('ica, iba -> icb', z_t, exptG12)
        
        # Reshape output into self.c*latent_dim channels:
        z_t_flat = z_t.view(-1, self.c*self.latent_dim)
        x_t = self.decoder(z_t_flat)
        
        ### PLOTTING FOR Z-TESTING ###
        # The following code is the z-test. It plots the latent vector z 
        # and the transformed latent vector z_t as latent_dim by latent_dim 
        # images. It also plots the flow of the generator G by plotting its
        # action on a two-dimensional gaussian blob. Also, arrows are plotted
        # that correspond to the flow of the generator G (by using the vector
        # field form of the generator G) on the gaussian blob. 

        if zTest and self.i < 1:
            # Plot the generator G as a vector field. The coefficients come from
            # a_norm and correspond to [constant,x,y] in the x-direction and
            # [constant,x,y] in the y-direction, respectively.

            x_plot = np.linspace(-5,5,10)
            y_plot = np.linspace(-5,5,10)
            X,Y = np.meshgrid(x_plot,y_plot)

            # Swap coordinates and make y -> -y to match image coordinates:
            X,Y = Y,X
            Y = -Y

            # Extract coefficients for G (a_normed) and G2 (a2_normed)
            a_values = [a_normed, a2_normed]
            titles = ["G (from a_normed)", "G2 (from a2_normed)"]

            for idx, a_vector in enumerate(a_values):
                # Extract coefficients for the current a vector
                a0 = a_vector[0].detach().cpu().numpy()
                a1 = a_vector[1].detach().cpu().numpy()
                a2 = a_vector[2].detach().cpu().numpy()
                a3 = a_vector[3].detach().cpu().numpy()
                a4 = a_vector[4].detach().cpu().numpy()
                a5 = a_vector[5].detach().cpu().numpy()

                # Print drift and diffusion terms for debugging
                print(f"Drift terms for {titles[idx]}:\n"
                    f"a0 = {a0}, a3 = {a3}\n"
                    f"Diffusion matrix:\n[{a1}, {a2}]\n[{a4}, {a5}]")

                # Calculate the vector field
                U = a0 + a1 * X + a2 * Y
                V = a3 + a4 * X + a5 * Y

                # Plot the vector field
                plt.figure(figsize=(6, 6))
                plt.quiver(X, Y, U, V, color='r')
                plt.title(f"Vector Field for {titles[idx]}")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.grid()
                plt.show()

            self.plot_patches(z,z_t,t)

            self.i += 1
        ################################    
        x_t = x_t.view(data_shape)

        return x_t, exptG, out_t

    def normal(self, x):
        # Save size before flattening input:
        data_shape = x.size()
        
        # Put x in correct shape:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x = self.decoder(z)

        # Unflattens output so it has same shape as input:
        x = x.view(data_shape)

        return x
    
    def generate_basis(self, d_squared, non_affine=False):
        L0 = lambda d,z: np.sum([2*np.pi*p/d**2 * np.sin(2*np.pi*p/d *z) 
                                 for p in np.arange(-d/2+1,d/2)], axis=0)
        # Latent dimension:
        d = int(np.sqrt(d_squared))

        coords = np.mgrid[:d,:d] - d/2
        x,y = coords.reshape((2,-1))

        # Swap coordinates and make y -> -y to match image coordinates:
        x,y = y,x
        y = -y

        dx = (x[:,np.newaxis] - x) * (y[:,np.newaxis] == y)
        dy = (y[:,np.newaxis] - y) * (x[:,np.newaxis] == x)

        Lx = L0(2*d, dx)
        Ly = L0(2*d, dy)
    
        xLx = np.diag(x) @ Lx
        yLx = np.diag(y) @ Lx
        xLy = np.diag(x) @ Ly
        yLy = np.diag(y) @ Ly

        if non_affine:
            # Do the same for x_squared and y_squared, and x*y:
            x_squared = np.diag(x)**2
            y_squared = np.diag(y)**2
            xy = np.diag(x) @ np.diag(y)
            
            x2Lx = x_squared @ Lx
            y2Lx = y_squared @ Lx
            xyLx = xy @ Lx
            x2Ly = x_squared @ Ly
            y2Ly = y_squared @ Ly
            xyLy = xy @ Ly

            D = np.stack([Lx, xLx, yLx, x2Lx,  xyLx, y2Lx, Ly, xLy, yLy, x2Ly,  xyLy, y2Ly], axis=0)
            D = torch.from_numpy(D).float()
        else:
            D = np.stack([Lx, xLx, yLx, Ly, xLy, yLy], axis=0)
            D = torch.from_numpy(D).float()

        return D
    
    def taylor_loss(self, combined_data, order=1):  
        # Normalize a vector by dividing by its norm:
        a_norm = torch.norm(self.a, p=2)
        a_normed = self.a / a_norm

        a2_norm = torch.norm(self.a2, p=2)
        a2_normed = self.a2 / a2_norm
        

        self.G1 = torch.einsum('i, imn -> mn', a_normed, self.D_big)
        self.G2 = torch.einsum('i, imn -> mn', a2_normed, self.D_big)
        commutator = torch.matmul(self.G1, self.G2) - torch.matmul(self.G2, self.G1)

        x, y = torch.split(combined_data, 1, dim=1)

        # Save size before flattening input:
        data_shape = x.size()

        # Flatten x and y:
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)   

        # Get t from concatenating x and y along the feature dimension
        # and passing through tNet:
        t, t2, t12 = self.t(torch.cat((x,y), dim=1)).split(1, dim=1)
        # Flatten t:
        t, t2, t12 = t.squeeze(), t2.squeeze(), t12.squeeze()
        # t does not require gradient:
        t_no_grad = t.detach()
        t2_no_grad = t2.detach()
        t12_no_grad = t12.detach()
        # prod_no_grad = t_no_grad * t2_no_grad
        # tG = torch.einsum('i, mn -> imn', t_no_grad, self.G)
        # First-order Taylor approximation
        taylor_approx = (
            torch.eye(self.D_big.shape[1]).to(x.device) + 
            torch.einsum('i, mn -> imn', t_no_grad, self.G1) + 
            torch.einsum('i, mn -> imn', t2_no_grad, self.G2) + 
            torch.einsum('i, mn -> imn', t12_no_grad, commutator)
        )

        # Apply Taylor approximation
        x_t = torch.einsum('ia, iba -> ib', x, taylor_approx)

        # Return the transformed x_t and y for comparison
        return x_t, y
    
        # if order == 1:    
        #     # First order Taylor approximation:
        #     exptG_order = torch.eye(self.D_big.shape[1]).to("cuda:0") + tG
        # elif order == 2:
        #     # Second order Taylor approximation:
        #     exptG_order = torch.eye(self.D_big.shape[1]).to("cuda:0") + tG + 0.5*torch.einsum('imn, ink -> imk', tG, tG)
        # else:
        #     raise ValueError("Order must be 1 or 2.")

        # x_t = torch.einsum('ia, iba -> ib', x, exptG_order)

        # return x_t, y
    

    def plot_patches(self, z, z_t, t):
        latent_dim_sqrt = int(np.sqrt(self.latent_dim))
        print(z.shape, z_t.shape, t.shape)
        # Initialize the figure with 4 rows and 11 columns
        fig, axs = plt.subplots(6, 11, figsize=(20, 8))  # Adjusted for 11 columns

        # Loop over the first 11 values to display z and z_t patches
        for j in range(11):
            # Plot the z patch in the first row
            z_plot = z[j, 0].view(latent_dim_sqrt, latent_dim_sqrt)
            axs[0, j].imshow(z_plot.detach().cpu().numpy(), cmap='viridis')
            axs[0, j].axis('off')
            axs[0, j].set_title(f't = {t[j].item():.2f}', fontsize=8)

            # Plot the z_t patch in the second row
            z_t_plot = z_t[j, 0].view(latent_dim_sqrt, latent_dim_sqrt)
            axs[1, j].imshow(z_t_plot.detach().cpu().numpy(), cmap='viridis')
            axs[1, j].axis('off')

        # Generate quasi-equally spaced t values between -1 and 1, including 0
        t_values = torch.linspace(-0.5, 0.5, 11, device='cuda:0')  # 11 values including -1 and 1

        # Create a Gaussian blob in the center for the sample patch
        z_temp1 = self.create_gaussian_line_segment(stddev=0.2) #self.create_gaussian_blob(stddev=0.5)
        z_temp2 = self.create_gaussian_blob(stddev=0.5)

        a_test = torch.tensor([0,0,-1,0,1,0]).float().to("cuda:0")
        a_test_norm = torch.norm(a_test, p=2)
        a_test_normed = a_test / a_test_norm

        G_test = torch.einsum('i, imn -> mn', a_test_normed, self.D)

        a_model = self.a
        a_model_norm = torch.norm(a_model, p=2)
        a_model_normed = a_model / a_model_norm

        G_model = torch.einsum('i, imn -> mn', a_model_normed, self.D)

        # Loop to plot the transformed Gaussian blob for each t value
        for j in range(11):  # Loop over all 11 t values
            t_sample = t_values[j]
            

            # Apply matrix exponentiation for each t value and visualize the effect
            exptG = torch.matrix_exp(torch.einsum('i,mn -> imn', t_sample.view(1), G_test))
            exptG_model = torch.matrix_exp(torch.einsum('i,mn -> imn', t_sample.view(1), G_model))
            
            z_t_temp = torch.einsum('ia, iba -> ib', z_temp1, exptG)
            z_t_temp_model = torch.einsum('ia, iba -> ib', z_temp1, exptG_model)

            z_t_temp_plot = z_t_temp.view(latent_dim_sqrt, latent_dim_sqrt)
            z_t_temp_plot_model = z_t_temp_model.view(latent_dim_sqrt, latent_dim_sqrt)

            # Plot the blob transformed by model in row 3
            axs[2, j].imshow(z_t_temp_plot_model.detach().cpu().numpy(), cmap='viridis')
            axs[2, j].axis('off')

            # Plot the transformed blob in row 4
            axs[3, j].imshow(z_t_temp_plot.detach().cpu().numpy(), cmap='viridis')
            axs[3, j].axis('off')

            # Optionally label the t values for clarity above the original Gaussian blob
            axs[2, j].set_title(f't = {t_sample.item():.2f}', fontsize=8)

        # Add a horizontal line between Rows 2 and 3 for clarity
        for j in range(11):
            axs[2, j].axhline(y=-0.5, color='black', linewidth=1)

                # Loop to plot the transformed Gaussian blob for each t value
        for j in range(11):  # Loop over all 11 t values
            t_sample = t_values[j]

            # Apply matrix exponentiation for each t value and visualize the effect
            exptG = torch.matrix_exp(torch.einsum('i,mn -> imn', 10*t_sample.view(1), G_test))
            exptG_model = torch.matrix_exp(torch.einsum('i,mn -> imn', t_sample.view(1), G_model))

            z_t_temp = torch.einsum('ia, iba -> ib', z_temp2, exptG)
            z_t_temp_model = torch.einsum('ia, iba -> ib', z_temp2, exptG_model)

            z_t_temp_plot = z_t_temp.view(latent_dim_sqrt, latent_dim_sqrt)
            z_t_temp_plot_model = z_t_temp_model.view(latent_dim_sqrt, latent_dim_sqrt)

            # Plot the original Gaussian blob in row 3
            axs[4, j].imshow(z_t_temp_plot_model.detach().cpu().numpy(), cmap='viridis')
            axs[4, j].axis('off')

            # Plot the transformed blob in row 4
            axs[5, j].imshow(z_t_temp_plot.detach().cpu().numpy(), cmap='viridis')
            axs[5, j].axis('off')

            # Optionally label the t values for clarity above the original Gaussian blob
            axs[4, j].set_title(f't = {t_sample.item():.2f}', fontsize=8)

        # Add a horizontal line between Rows 2 and 3 for clarity
        for j in range(11):
            axs[4, j].axhline(y=-0.5, color='black', linewidth=1)


        plt.tight_layout()
        plt.show()


    def create_gaussian_blob(self, stddev=0.5):
        # Create a 2D grid of coordinates
        latent_dim_sqrt = int(np.sqrt(self.latent_dim))
        x = torch.linspace(-1, 1, latent_dim_sqrt)
        y = torch.linspace(-1, 1, latent_dim_sqrt)
        X, Y = torch.meshgrid(x, y)

        # Calculate the 2D Gaussian function
        gaussian_blob = torch.exp(-(((X)**2 + (Y)**2) / (2 * stddev**2)))
        # Rotate the Gaussian blob by 90 degrees

        gaussian_blob= torch.rot90(gaussian_blob,k=1)

        # Flatten the Gaussian blob to match the latent vector dimensions
        gaussian_blob = gaussian_blob.reshape(1, self.latent_dim).to("cuda:0")
        return gaussian_blob
    
    def create_gaussian_line_segment(self, stddev=0.5):
        # Create a 2D grid of coordinates
        latent_dim_sqrt = int(np.sqrt(self.latent_dim))
        x = torch.linspace(-1, 1, latent_dim_sqrt)
        y = torch.linspace(-1, 1, latent_dim_sqrt)
        X, Y = torch.meshgrid(x, y)

        # Calculate the 2D Gaussian line
        gaussian_blob = torch.exp(-((X**2 ) / (2 * stddev**2)))
        
        # Set the first and last row to zero
        gaussian_blob[:, 0] = 0
        gaussian_blob[:, -1] = 0

        # Flatten the Gaussian blob to match the latent vector dimensions
        gaussian_blob = gaussian_blob.reshape(1, self.latent_dim).to("cuda:0")
        return gaussian_blob
    
    def create_L_shape(length=5, width=1):
        """
        Create an L-shaped figure.
        
        Parameters:
        - length: Length of the arms of the L shape.
        - width: Width of the arms of the L shape.
        
        Returns:
        A tensor representing the L-shaped figure.
        """
        L_shape = torch.zeros((1, length, length)).to("cuda:0")  # Create a blank square tensor
        L_shape[0, :length, :width] = 1  # Vertical arm
        L_shape[0, :width, :length] = 1  # Horizontal arm
        return L_shape


class EncoderLieMulTVecDecoder(BaseModel):
    def __init__(self, input_size, hidden_sizes, t_hidden_sizes, latent_dim, non_affine=False,channels=4, dropout=False):
        super().__init__()
        self.latent_dim = latent_dim  # Dimensionality of the latent vector
        self.c = channels

        # Encoder: Produces a latent vector z_0
        self.encoder = Encoder(input_size, hidden_sizes, channels * latent_dim, layer_norm=False)

        # Trainable transformation matrix G
        self.a = nn.Parameter(torch.randn(latent_dim, latent_dim), requires_grad=True)

        # Decoder: Reconstructs the output from z_t
        if dropout:
            self.decoder = DecoderDropout(channels * latent_dim, hidden_sizes, input_size, dropout=0.5)
        else:
            self.decoder = Decoder(channels * latent_dim, hidden_sizes, input_size, layer_norm=False)

        # tNet: Predicts transformation magnitudes t
        self.t = tNet(2 * input_size, t_hidden_sizes, tnum=1)  # Only one t value

        self.i = 0

    def forward(self, x, epsilon=0, tMax=0, zTest=False):
        # Normalize G
        a_norm = torch.norm(self.a, p="fro")
        a_normalized = self.a / a_norm

        # Split input into x (input) and y (target)
        x, y = torch.split(x, 1, dim=1)
        data_shape = x.size()
        # Flatten x and y for processing
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        # Determine t
        if epsilon == 0 and tMax == 0:
            out_t = self.t(torch.cat((x, y), dim=1))
            t = out_t.squeeze(1)
        elif epsilon != 0 and tMax == 0:
            t = epsilon * torch.ones(x.size(0)).to(x.device)
        elif tMax > 0 and epsilon == 0:
            t = torch.linspace(-tMax, tMax, x.size(0)).to(x.device)

        # Matrix exponential for transformation
        exponent = torch.einsum("i,mn->imn", t, a_normalized)
        exptG = torch.matrix_exp(exponent)

        # Encode to z_0 (latent vector) and transform to z_t
        z = self.encoder(x).view(-1, self.c, self.latent_dim)  # Reshape to (batch, channels, latent_dim)
        z_t = torch.einsum("ica,iba->icb", z, exptG)  # Apply transformation

        # Decode z_t back to the original input space
        z_t_flat = z_t.view(-1, self.c * self.latent_dim)
        x_t = self.decoder(z_t_flat)

        # Optionally visualize transformations during testing
        if zTest and self.i < 1:
            self.visualize(z, z_t, t, a_normalized)
            self.i += 1

        # Reshape output back to the original input size
        x_t = x_t.view(data_shape)
        return x_t, exptG, t

    def normal(self, x):
        # Save size before flattening input:
        data_shape = x.size()
        
        # Put x in correct shape:
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x = self.decoder(z)

        # Unflattens output so it has same shape as input:
        x = x.view(data_shape)

        return x

    def visualize(self, z, z_t, t, a_normalized):
        """
        Visualize the original latent vectors (z) and transformed latent vectors (z_t) in 2D.
        """
        z = z.view(-1, self.latent_dim).detach().cpu().numpy()  # Flatten and move to CPU
        z_t = z_t.view(-1, self.latent_dim).detach().cpu().numpy()  # Flatten and move to CPU

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Plot original latent vectors
        plt.scatter(z[:, 0], z[:, 1], c="blue", alpha=0.6, label="Original Latents (z)")

        # Plot transformed latent vectors
        plt.scatter(z_t[:, 0], z_t[:, 1], c="orange", alpha=0.6, label="Transformed Latents (z_t)")

        # Add labels and legend
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title("Latent Space Visualization")
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.show()


# class EncoderMultiLieDecoder(BaseModel):
#     def __init__(self, input_size, hidden_sizes, t_hidden_sizes, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.encoder = Encoder(input_size, hidden_sizes, latent_dim)

#         # Initialize 6 trainable parameters for basis coefficients:
#         self.a = nn.Parameter(torch.randn(6), requires_grad=True)
#         # self.a2 = nn.Parameter(torch.randn(6), requires_grad=True)


#         self.D = self.generate_basis(d_squared=self.latent_dim).clone().detach().to("cuda:0")
#         self.D_big = self.generate_basis(d_squared=input_size).clone().detach().to("cuda:0")

#         # self.D2 = self.generate_basis(d_squared=self.latent_dim).clone().detach().to("cuda:0")
#         # self.D_big2 = self.generate_basis(d_squared=input_size).clone().detach().to("cuda:0")

#         self.decoder = Decoder(latent_dim, hidden_sizes, input_size)
#         self.t = tNet(2*input_size, t_hidden_sizes)

#         self.i = 0

#     def forward(self, x, epsilon=0, tMax=0, zTest=False):
#         # Normalize a vector by dividing by its norm:
#         a_norm = torch.norm(self.a, p=2)
#         a_normed = self.a / a_norm
#         # a2_norm = torch.norm(self.a2, p=2)
#         # a2_normed = self.a2 / a2_norm

#         # a_tanh = self.tanh(self.a) # Squash to [-1,1]
#         self.G = torch.einsum('i, imn -> mn', a_normed, self.D)
#         # self.G2 = torch.einsum('i, imn -> mn', a2_normed, self.D2)

#         x, y = torch.split(x, 1, dim=1)

#         # Save size before flattening input:
#         data_shape = x.size()

#         # Flatten x and y:
#         x = x.view(x.size(0), -1)
#         y = y.view(y.size(0), -1)   
        
#         # For zero epsilon, use the tNet, else set t to epsilon:
#         if epsilon == 0 and tMax == 0:
#             # Get t from concatenating x and y along the feature dimension
#             # and passing through tNet:
#             t = self.t(torch.cat((x,y), dim=1))
#             # Split t into two parts, one for each output neuron and squeeze seperately:
#             # t, t2 = torch.split(t, 1, dim=1)
#             t = t.squeeze()
#             # t2 = t2.squeeze()

#         elif epsilon != 0 and tMax == 0 and self.i < 1:
#             t = epsilon * torch.ones(x.size(0)).to("cuda:0")
#         elif tMax > 0 and epsilon == 0 and self.i < 1:
#             # tMax sets the maximum value of t in the interval [-tMax, tMax]
#             # in which the number of steps agrees with the batch size:
#             t = torch.linspace(-tMax, tMax, x.size(0)).to("cuda:0")
        
#         # Flattens input:
#         z = self.encoder(x)
#         exponent = torch.einsum('i, mn -> imn', t, self.G)
#         # exponent2 = torch.einsum('i, mn -> imn', t2, self.G2)

#         exptG = torch.matrix_exp(exponent)
#         # exptG2 = torch.matrix_exp(exponent2)

#         z_t = torch.einsum('ia, iba -> ib', z, exptG)
#         # z_t2 = torch.einsum('ia, iba -> ib', z_t, exptG2)

#         x_t = self.decoder(z_t)
        
#         ### PLOTTING FOR Z-TESTING ###
#         # The following code is the z-test. It plots the latent vector z 
#         # and the transformed latent vector z_t as latent_dim by latent_dim 
#         # images. It also plots the flow of the generator G by plotting its
#         # action on a two-dimensional gaussian blob. Also, arrows are plotted
#         # that correspond to the flow of the generator G (by using the vector
#         # field form of the generator G) on the gaussian blob. 

#         if zTest and self.i < 1:
#             # Plot the generator G as a vector field. The coefficients come from
#             # a_norm and correspond to [constant,x,y] in the x-direction and
#             # [constant,x,y] in the y-direction, respectively.

#             x_plot = np.linspace(-5,5,10)
#             y_plot = np.linspace(-5,5,10)
#             X,Y = np.meshgrid(x_plot,y_plot)
            
#             # Get a0, a1, a2, a3, a4, a5 from a_normed and put on the cpu and print:
#             a0 = a_normed[0].detach().cpu().numpy()
#             a1 = a_normed[1].detach().cpu().numpy()
#             a2 = a_normed[2].detach().cpu().numpy()
#             a3 = a_normed[3].detach().cpu().numpy()
#             a4 = a_normed[4].detach().cpu().numpy()
#             a5 = a_normed[5].detach().cpu().numpy()
#             print("2a = ", a0, a1, a2, a3, a4, a5)

#             # Plot the vector field:
#             # x components are a0*1 + a1*x + a2*y, y components are a3*1 + a4*x + a5*y:
#             U = a0 + a1*X + a2*Y
#             V = a3 + a4*X + a5*Y
#             plt.quiver(X, Y, U, V, color='r')
#             plt.grid()
#             plt.show()
#             # Plot z and z_t as images:
#             for j in range(10):
#                 print('t:', t[j])
#                 z_plot = z[j].view(-1, int(np.sqrt(self.latent_dim)), int(np.sqrt(self.latent_dim)))
#                 z_t_plot = z_t[j].view(-1, int(np.sqrt(self.latent_dim)), int(np.sqrt(self.latent_dim)))
#                 fig, ax = plt.subplots(2,2)
#                 ax[0].imshow(z_plot.detach().cpu().numpy()[0])
#                 ax[1].imshow(z_t_plot.detach().cpu().numpy()[0])
#                 # MAke a matrix with zeros and a single 1 in the middle:
#                 z_temp = torch.zeros((1, self.latent_dim)).to("cuda:0")
#                 z_temp[0, int(self.latent_dim/2)] = 1
#                 z_temp_plot = z_temp.view(-1, int(np.sqrt(self.latent_dim)), int(np.sqrt(self.latent_dim)))
#                 # Now plot the effect of G on the gaussian blob:
#                 z_t_temp = torch.einsum('ia, iba -> ib', z_temp, exptG)
#                 z_t_temp_plot = z_t_temp.view(-1, int(np.sqrt(self.latent_dim)), int(np.sqrt(self.latent_dim)))
#                 ax[2].imshow(z_t_temp_plot.detach().cpu().numpy()[0])
#                 ax[3].imshow(z_temp_plot.detach().cpu().numpy()[0])
#                 plt.show()
                


#             self.i += 1
#         ################################    

#         x_t = x_t.view(data_shape)

#         return x_t, t

#     def normal(self, x):
#         # Save size before flattening input:
#         data_shape = x.size()

#         # Flattens input:
#         x = x.view(x.size(0), -1)
#         z = self.encoder(x)
#         x = self.decoder(z)

#         # Unflattens output so it has same shape as input:
#         x = x.view(data_shape)
#         return x
    
#     def generate_basis(self, d_squared):
#         L0 = lambda d,z: np.sum([2*np.pi*p/d**2 * np.sin(2*np.pi*p/d *z) 
#                                  for p in np.arange(-d/2+1,d/2)], axis=0)
#         # Latent dimension:
#         d = int(np.sqrt(d_squared))

#         coords = np.mgrid[:d,:d] - d/2
#         x,y = coords.reshape((2,-1))

#         dx = (x[:,np.newaxis] - x) * (y[:,np.newaxis] == y)
#         dy = (y[:,np.newaxis] - y) * (x[:,np.newaxis] == x)

#         Lx = L0(2*d, dx)
#         Ly = L0(2*d, dy)
    
#         xLx = np.diag(x) @ Lx
#         yLx = np.diag(y) @ Lx
#         xLy = np.diag(x) @ Ly
#         yLy = np.diag(y) @ Ly

#         D = np.stack([Lx, xLx, yLx, Ly, xLy, yLy], axis=0)
#         D = torch.from_numpy(D).float()
#         return D
    
#     def taylor_loss(self, combined_data, order=1):
#         # Normalize a vector by dividing by its norm:
#         a_norm = torch.norm(self.a, p=2)
#         a_normed = self.a / a_norm

#         # a_tanh = self.tanh(self.a) # Squash to [-1,1]

#         self.G_big = torch.einsum('i, imn -> mn', a_normed, self.D_big)

#         x, y = torch.split(combined_data, 1, dim=1)

#         # Save size before flattening input:
#         data_shape = x.size()

#         # Flatten x and y:
#         x = x.view(x.size(0), -1)
#         y = y.view(y.size(0), -1)   

#         # Get t from concatenating x and y along the feature dimension
#         # and passing through tNet:
#         t = self.t(torch.cat((x,y), dim=1))
#         # Flatten t:
#         t = t.squeeze()
#         # t does not require gradient:
#         t_no_grad = t.detach()

#         tG = torch.einsum('i, mn -> imn', t_no_grad, self.G_big)

#         if order == 1:    
#             # First order Taylor approximation:
#             exptG_order = torch.eye(self.D_big.shape[1]).to("cuda:0") + tG
#         elif order == 2:
#             # Second order Taylor approximation:
#             exptG_order = torch.eye(self.D_big.shape[1]).to("cuda:0") + tG + 0.5*torch.einsum('imn, ink -> imk', tG, tG)
#         else:
#             raise ValueError("Order must be 1 or 2.")

#         x_t = torch.einsum('ia, iba -> ib', x, exptG_order)

#         return x_t, y
