import torch
import torch.nn as nn
import torch.nn.functional as F


class DRAW(nn.Module):
    def __init__(self, image_size, h_dim, z_dim, T, N, epsilon=1e-8):
        super().__init__()
        self.image_size = image_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.T = T
        self.N = N
        self.epsilon = torch.tensor(epsilon)
        self.d_image = image_size[0] * image_size[1] * image_size[2]

        if N:
            self.read = self.read_with_attention
            self.write = self.write_with_attention
            # patch dimension: channels * N * N
            if isinstance(N, int):
                N_r = N_w = N
            else:
                N_r, N_w = N

            self.patch_size_r = image_size[0] * N_r * N_r
            self.patch_size_w = image_size[0] * N_w * N_w
        else:
            self.read = self.read_no_attention
            self.write = self.write_no_attention
            # patch dimension is same as input image
            self.patch_size_r = self.d_image
            self.patch_size_w = self.d_image

        self.encode = nn.GRUCell(2 * self.patch_size_r + h_dim, h_dim)
        self.decode = nn.GRUCell(z_dim, h_dim)

        self.mu_linear = nn.Linear(h_dim, z_dim)
        self.log_sigma_linear = nn.Linear(h_dim, z_dim)
        self.write_linear = nn.Linear(h_dim, self.patch_size_w)

        # 5 params: g~_x, g~_y, log(σ), log(𝛿~), log(γ)
        self.read_attn = nn.Linear(h_dim, 5)
        self.write_attn = nn.Linear(h_dim, 5)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        reconsts = torch.zeros_like(x)
        h_enc = torch.zeros(*[x.shape[0], self.h_dim], device=x.device)
        h_dec = torch.zeros(*[x.shape[0], self.h_dim], device=x.device)

        # mus ans sigmas of every steps are needed to compute latent loss L_z
        self.mus = []
        self.sigmas = []
        self.attn_deltas = []
        self.attn_sigmas = []
        self.attn_gx_read = []
        self.attn_gx_write = []
        self.attn_gy_read = []
        self.attn_gy_write = []

        for step in range(self.T):
            x_hat = x - torch.sigmoid(reconsts[-1])
            r = self.read(x, x_hat, h_dec)
            h_enc = self.encode(torch.hstack([r, h_dec]), h_enc)
            z = self.sample_from_Q(x, h_enc, step)
            h_dec = self.decode(z, h_dec)
            reconsts = reconsts + self.write(h_dec)

        self.mus = torch.stack(self.mus)
        self.sigmas = torch.stack(self.sigmas)

        return torch.sigmoid(reconsts).reshape(-1, *self.image_size)

    def read_no_attention(self, x, x_hat, h_dec):
        return torch.hstack([x, x_hat])

    def read_with_attention(self, x, x_hat, h_dec):
        channels, A, B = self.image_size
        x_shape = (x.shape[0] * channels, A, B)

        N = self.N if isinstance(self.N, int) else self.N[0]
        F_x, F_y, gamma = self.get_filterbanks(
            h_dec, self.read_attn, N, is_read=True)
        F_x.transpose_(-1, -2)

        x_attn = self.compute_read_attn(F_x, F_y, x.reshape(x_shape))
        x_hat_attn = self.compute_read_attn(F_x, F_y, x_hat.reshape(x_shape))

        return gamma * torch.hstack([x_attn, x_hat_attn])

    def compute_read_attn(self, F_x_t, F_y, img):
        return torch.reshape(F_y @ img @ F_x_t, [img.shape[0], -1])

    def write_no_attention(self, h_dec):
        return self.write_linear(h_dec)

    def write_with_attention(self, h_dec):
        num_channels, A, B = self.image_size

        N = self.N if isinstance(self.N, int) else self.N[1]
        F_x, F_y, gamma = self.get_filterbanks(
            h_dec, self.write_attn, N, is_read=False)
        F_x = torch.vstack([F_x] * num_channels)
        F_y = torch.vstack([F_y] * num_channels)

        w = self.write_linear(h_dec)
        w = w.reshape(-1, N, N)

        Fy_w_Fx = (F_y.transpose(-1, -2) @ w @ F_x).reshape(h_dec.shape[0], -1)
        return Fy_w_Fx / gamma

    def get_filterbanks(self, h_dec, W, N, is_read):
        device = h_dec.device
        _, A, B = self.image_size

        # 5 params: g~_x, g~_y, log(σ^2), log(𝛿~), log(γ)
        gt_x, gt_y, log_sigma_sq, log_delta_tilde, log_gamma = \
            W(h_dec).split(1, dim=1)

        # transform params emitted in log-scale to ensure positivity
        two_sigma_sq = torch.reshape(2 * torch.exp(log_sigma_sq), [-1, 1, 1])
        delta = (max(A, B) - 1) * torch.exp(log_delta_tilde) / (N - 1)
        gamma = torch.exp(log_gamma)

        # compute filterbanks
        g_x, F_x = self.compute_filterbank(
            gt_x, delta, two_sigma_sq, A, N, device)
        g_y, F_y = self.compute_filterbank(
            gt_y, delta, two_sigma_sq, B, N, device)

        # self.attn_deltas.append(delta)
        self.attn_deltas.append(delta * (N - 1))
        self.attn_sigmas.append((two_sigma_sq / 2)**0.5)
        if is_read:
            self.attn_gx_read.append(g_x)
            self.attn_gy_read.append(g_y)
        else:
            self.attn_gx_write.append(g_x)
            self.attn_gy_write.append(g_y)

        return F_x, F_y, gamma

    def compute_filterbank(self, gt, delta, two_sigma_sq, num_pixels, N, device):
        # grid and pixel range
        grid_range = torch.arange(N, device=device)
        pixel_range = torch.arange(num_pixels, device=device)
        pixel_range = pixel_range.reshape(1, 1, num_pixels)

        # From center of attention `g`, compute mean locations `mu` of filters.
        # Then, use these to compute normalized filterbank matrix `F`.
        g = 0.5 * (num_pixels + 1) * (gt + 1)
        mu = g + (grid_range - 0.5 * (N - 1)) * delta
        mu = mu.reshape(-1, N, 1)
        F = torch.exp(-(pixel_range - mu)**2 / two_sigma_sq)
        return g, F / torch.maximum(F.sum(dim=-1, keepdim=True), self.epsilon)

    def sample_from_P(self, num_samples, device):
        return torch.randn(*[num_samples, self.z_dim], device=device)

    def sample_from_Q(self, x, h_enc, step):
        # need to store those values to compute latent loss L_z
        self.mus.append(self.mu_linear(h_enc))
        self.sigmas.append(torch.exp(self.log_sigma_linear(h_enc)))

        # reparameterization trick
        samples = torch.randn(*[x.shape[0], self.z_dim], device=x.device)

        return self.mus[-1] + samples * self.sigmas[-1]

    def generate_images(self, num_images, device):
        # set a fixed seed to ensure consistency between epochs
        saved_rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        self.attn_deltas = []
        self.attn_sigmas = []
        self.attn_gx_write = []
        self.attn_gy_write = []

        with torch.no_grad():
            reconsts = [torch.zeros(num_images, self.d_image).to(device)]
            h_dec = torch.zeros(*[num_images, self.h_dim], device=device)

            for _ in range(self.T):
                z = self.sample_from_P(num_images, device)
                h_dec = self.decode(z, h_dec)
                reconsts.append(reconsts[-1] + self.write(h_dec))

        # restore saved RNG state
        torch.set_rng_state(saved_rng_state)

        # we can drop first step as it is completely grey
        return torch.sigmoid(torch.stack(reconsts[1:]))

    def compute_loss(self, image, reconst, reduction='mean'):
        reconst_loss = F.binary_cross_entropy(reconst, image, reduction='none')
        reconst_loss = reconst_loss.reshape(image.shape[0], -1)
        reconst_loss = torch.sum(reconst_loss, dim=-1)

        mus_sq = self.mus * self.mus
        sigmas_sq = self.sigmas * self.sigmas
        log_simgas_sq = torch.log(sigmas_sq + 1e-10)
        latent_loss = torch.sum(mus_sq + sigmas_sq - log_simgas_sq, dim=0)
        latent_loss = torch.sum(0.5 * (latent_loss - self.T), dim=-1)

        total_loss = torch.mean(reconst_loss + latent_loss)

        # print("reconst_loss", reconst_loss.mean().detach().item())
        # print("latent_loss", latent_loss.mean().detach().item())

        # if reduction == 'sum':
        #     total_loss = total_loss.sum()
        # elif reduction == 'mean':
        #     total_loss = total_loss.mean()

        return total_loss
