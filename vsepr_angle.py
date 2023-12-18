import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def main(args):
    n_point = args.n_atom
    points = torch.randn(n_point-1, 3, requires_grad=True)
    fixed_point = torch.tensor([[0., 0., 1.]])

    optimizer = torch.optim.AdamW([points], lr=1e-3)
    tqdm_loader = tqdm(range(n_point*3000))
    for _ in tqdm_loader:
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
    ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax)  # IMPORTANT - this is also required
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_atom', default=6, type=int)
    arg = parser.parse_args()
    main(arg)
