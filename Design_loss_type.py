import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
z_all = np.array([
    [[0.5, 0.5],
     [1.5, 0.5],
     [1.5, 1.5],
     [0.5, 1.5]],
    [[0, -1],
     [2, -1],
     [2, 1],
     [0, 1]],
    [[-1, -1],
     [1, -1],
     [1, 1],
     [-1, 1]]
])
z_all = z_all.transpose(0, 2, 1)
print(z_all.shape)
z_all_torch = torch.from_numpy(z_all).float()
z_all_mean = torch.mean(z_all_torch, dim=2)
print(z_all_torch)
print(z_all_mean)
z_all_centered = z_all_torch - z_all_mean.unsqueeze(2)
z_all_dist = torch.sqrt(torch.sum(z_all_centered ** 2, dim=1))
print(z_all_dist)
z_all_dist_mean = torch.mean(z_all_dist, dim=1)

z_all_mean_np = z_all_mean.numpy()
z_all_dist_np = z_all_dist.numpy()
z_all_dist_mean_np = z_all_dist_mean.numpy()

plt.figure()
plt.scatter([0], [0], color="#FF0000")
for i, z_mat in enumerate(z_all):
    plt.scatter(z_mat.T[:, 0], z_mat.T[:, 1], color=cm.tab10(i), marker="x", alpha=1)
    for z, dist in zip(z_mat.T, z_all_dist_np[i, :]):
        plt.text(z[0], z[1], f"D = {dist:.3f}")

    plt.scatter(z_all_mean_np[i, 0], z_all_mean_np[i, 1], color=cm.tab10(i))
    plt.text(z_all_mean_np[i, 0], z_all_mean_np[i, 1], f"{z_all_dist_mean_np[i]:.3f}", color=cm.tab10(i))
plt.gca().set_aspect("equal")
plt.show()
