import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse


def main(args):
    n_point = args.n_atom
    points = torch.randn(n_point-1, 3, requires_grad=True)
    fixed_point = torch.tensor([[0., 0., 1.]])

    optimizer = torch.optim.AdamW([points], lr=1e-3)
    tqdm_loader = tqdm(range(n_point*3000))
    for i in tqdm_loader:
        optimizer.zero_grad()
        d = torch.cat([fixed_point, F.normalize(points, dim=1)], dim=0)
        d = d @ d.T
        d.fill_diagonal_(-1e10)
        loss = (1/(2-2*d)).sum()
        loss.backward()
        optimizer.step()
        tqdm_loader.set_description(f'loss={loss.item():.6f}')

    points = torch.cat([fixed_point, F.normalize(points, dim=1)], dim=0).detach().numpy()
    angles = np.arccos(points@points.T)*180/np.pi
    np.fill_diagonal(angles, 360)
    print('Minimum bond angle: ', angles.min())
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', c='b')
    ax.scatter(0, 0, 0, c='r', marker='o')
    for p in points:
        ax.plot([0, p[0]], [0, p[1]], zs=[0, p[2]], c='k')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_atom', default=6, type=int)
    arg = parser.parse_args()
    main(arg)
