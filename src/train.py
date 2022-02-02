import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from draw import DRAW
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

if __name__ == '__main__':
    valid_size = 5000
    num_epochs = 100
    batch_size = 2048

    base_lr = 5e-4
    lr = base_lr * batch_size / 256

    torch.manual_seed(42)

    kwargs = dict(root='../../data', download=True, transform=ToTensor())
    train = datasets.MNIST(train=True, **kwargs)
    test = datasets.MNIST(train=False, **kwargs)

    train, valid = random_split(train, [len(train) - valid_size, valid_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    del train, valid, test

    N = (2, 5)
    T = 64
    IMAGE_SIZE = train_loader.dataset[0][0].shape
    model = DRAW(image_size=IMAGE_SIZE, h_dim=256, z_dim=100, T=T, N=N)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_fn = DrawLoss(reduction='none')

    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(1, num_epochs + 1)):
            epoch_losses = []

            for batch, _ in train_loader:
                # send batch to device
                batch = batch.to(device)

                # forward pass
                reconst = model(batch)
                # loss = loss_fn(batch, reconst, model.mus, model.sigmas)
                loss = model.compute_loss(batch, reconst)

                # backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_losses.append(loss.detach().cpu().item())

            print(sum(epoch_losses) / len(train_loader))

            images = model.generate_images(
                num_images=8, device=device).detach()

            one_every = T // 16
            images = images[one_every - 1::one_every]

            images = rearrange(images, 'step b (c h w) -> (b h) (step w) c',
                               **dict(zip(['c', 'h', 'w'], IMAGE_SIZE)))
            images = images.squeeze().cpu().numpy()

            fig, ax = plt.subplots()
            ax.imshow(images, cmap='gray', vmin=0.0, vmax=1.0)

            for i in range(len(model.attn_deltas[0])):
                for t in range(len(model.attn_deltas)):
                    if t % one_every != one_every - 1:
                        continue
                    delta = model.attn_deltas[t][i]
                    sigma = model.attn_sigmas[t][i]
                    gx = model.attn_gx_write[t][i]
                    gy = model.attn_gy_write[t][i]
                    t = t // one_every
                    box = patches.Rectangle(
                        (gx - delta / 2 + 28 * t, gy - delta / 2 + 28 * i),
                        delta, delta,
                        linewidth=sigma**0.5, edgecolor='r', facecolor='none')
                    ax.add_patch(box)

            plt.title(f'Epoch {epoch}')
            plt.xlabel('steps')
            plt.xticks(torch.arange(T // one_every) * 28 + 28 // 2,
                       range(one_every, T + 1, one_every))
            plt.yticks([])
            plt.savefig(f'epoch_{epoch}.png')
            plt.close('all')
