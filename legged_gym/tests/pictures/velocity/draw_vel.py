# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
import os

VEL_DIR = os.path.dirname(os.path.realpath(__file__)) 
os.chdir(VEL_DIR)
fault_group_num = 4

TRUN_TO = "right"
TRUN_TO = "left"
TRUN_TO = "forward"
env_group_idx = [0, 2, 3]
env_group_name = ["health", "weak", "limit"]
colors_group = ["blue", "green", "orange"]

buf_name_list = ["command_x",  "base_vel_x", "command_yaw", "base_vel_yaw"]
if (TRUN_TO == "left" or TRUN_TO == "right"):
    for i in range(len(buf_name_list)):
        buf_name_list[i] += f"_{TRUN_TO}"
x_com_buffer = torch.load(f"./{buf_name_list[0]}.pt")
x_vel_buffer = torch.load(f"./{buf_name_list[1]}.pt")
yaw_com_buffer = torch.load(f"./{buf_name_list[2]}.pt")
yaw_vel_buffer = torch.load(f"./{buf_name_list[3]}.pt")

nb_rows, nb_cols = (1, 2)
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(8.3, 4))

a = axs[0]
time = np.linspace(0, len(x_com_buffer)*0.02, len(x_com_buffer))
collections = torch.zeros((6,2))

pad = 20
for i in range(3):
    a.plot(time, x_vel_buffer[:, env_group_idx[i]], c=colors_group[i], label=f'{env_group_name[i]} measured')
    print(f"{f'{env_group_name[i]} x:':>{pad}} mean {'%.3f'%torch.mean(x_vel_buffer[:, env_group_idx[i]])} var {'%.3f'%torch.var(x_vel_buffer[:, env_group_idx[i]])}")
    collections[i].copy_(torch.tensor((torch.mean(x_vel_buffer[:, env_group_idx[i]]), torch.var(x_vel_buffer[:, env_group_idx[i]]))))
a.plot(time, x_com_buffer[:, 0], c='red', label='command')
a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base linear velocity x')
a.legend(loc='lower right')

print("-" * 53)

a = axs[1]
time = np.linspace(0, len(yaw_com_buffer)*0.02, len(yaw_com_buffer))
for i in range(3):
    a.plot(time, yaw_vel_buffer[:, env_group_idx[i]], c=colors_group[i], label=f'{env_group_name[i]} measured')
    print(f"{f'{env_group_name[i]} yaw:':>{pad}} mean {'%.3f'%torch.mean(yaw_vel_buffer[:, env_group_idx[i]])} var {'%.3f'%torch.var(yaw_vel_buffer[:, env_group_idx[i]])}")
    collections[i+3].copy_(torch.tensor((torch.mean(x_vel_buffer[:, env_group_idx[i]]), 
                           torch.var(x_vel_buffer[:, env_group_idx[i]]))))
a.plot(time, yaw_com_buffer[:, 0], c='red', label='command')
a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base angular velocity yaw')
a.legend(loc='upper left')

print("-" * 53)
print(collections)

plt.tight_layout()
plt.savefig(f"./Agents-Vel_Tracking_{TRUN_TO}1.png", dpi=300)
plt.show()


# %%
pad = 20
print(f"{'Computation:':>{pad}}1")