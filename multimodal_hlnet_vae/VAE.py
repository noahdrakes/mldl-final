import torch
from torch import nn

class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_means, z_log_vars):
        epsilon = torch.randn_like(z_means, dtype=torch.float32)
        return z_means + torch.exp(0.5 * z_log_vars) * epsilon

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_dim=1152, seq_len=200):
        super().__init__()
        self.latent_dim = latent_dim

        # Reduced number of feature maps in encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, stride=2, padding=1),  # Reduced from 576
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),  # Reduced from 288
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),  # Reduced from 144
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Flatten()
        )

        flattened_dim = 64 * 25  # Updated based on reduced features

        self.lin_mean = nn.Sequential(
            nn.Linear(flattened_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.lin_log_var = nn.Sequential(
            nn.Linear(flattened_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )

        self.sampling = Sampling()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        z_means = self.lin_mean(x)
        z_log_vars = self.lin_log_var(x)
        z = self.sampling(z_means, z_log_vars)
        return z, z_means, z_log_vars

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim=1152, seq_len=200):
        super().__init__()
        self.seq_len = seq_len
        flattened_dim = 64 * 25  # Updated based on reduced features

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flattened_dim),
            nn.BatchNorm1d(flattened_dim),
            nn.ReLU(True)
        )

        # Reduced number of feature maps in decoder
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Reduced from 144->288
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Reduced from 288->576
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # Reduced from 576->input_dim
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_fc(x)
        x = x.view(-1, 64, 25)  # Updated based on reduced features
        x = self.decoder_conv(x)
        x = x.permute(0, 2, 1)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim=1152, seq_len=200):
        super().__init__()
        self.encoder = Encoder(latent_dim, input_dim, seq_len)
        self.decoder = Decoder(latent_dim, input_dim, seq_len)

    def forward(self, x):
        z, z_means, z_log_vars = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_means, z_log_vars
