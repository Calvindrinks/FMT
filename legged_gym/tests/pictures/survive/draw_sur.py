# %%
# Pictures
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

Test_env = ["health", "weak", "limit"]
colors = ["blue", "green", "orange"]
speed = ["middle", "high"]

SUR_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(SUR_DIR)

name_prefix = './sur_'
data_lists = []
exp_names = ["normal", "mixTrain_normal", "ours"]
exp_name = ["Normal", "Mix", "Ours"]

for i in range(3):
    for exp in exp_names:
        file_name = name_prefix + Test_env[i] + '_' + exp + ".pt"
        data_lists.append(torch.load(file_name).unsqueeze(0))

data_tensor = torch.cat(data_lists, dim=0)
data_tensor = data_tensor.view(3, 3, 2, -1)
print(f"data_tensor: {data_tensor.shape}")

speednum = 1
nb_rows, nb_cols = (speednum, 3)
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(8.3, 2.8))
for j in range(speednum):  # two speed group
    for i in range(len(Test_env)):
        if speednum is not 1:
            a = axs[j, i]
        else:
            a = axs[i]
        a.set(ylabel='survive rate [%]', title=Test_env[i].capitalize())
        bplot=a.boxplot(data_tensor[i, :, j], vert=True, patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

plt.setp(axs, xticks=[1,2,3],
         xticklabels=exp_name)
plt.tight_layout()
plt.savefig(f"./survive-compare1.png", dpi=300)
plt.show()


# %%
# Pictures
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

Test_env = ["weak", "limit"]
colors = [["green", "lightgreen"], ["blue", "cyan"]]
speed = ["middle", "high"]

SUR_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(SUR_DIR)

name_prefix = './sur_'
data_lists = []
exp_names = ["wo_reflect", "ours"]
exp_name = ["W/O Ref", "Ours"]

for i in range(2):
    for exp in exp_names:
        file_name = name_prefix + Test_env[i] + '_' + exp + ".pt"
        data_lists.append(torch.load(file_name).unsqueeze(0))

data_tensor = torch.cat(data_lists, dim=0)
data_tensor = data_tensor.view(2, 2, 2, -1)
print(f"data_tensor: {data_tensor.shape}")

nb_rows, nb_cols = (1, 4)
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(8.3, 2.5))
for j in range(2):  # two speed group
    for i in range(len(Test_env)):
        a = axs[j * 2 + i]
        a.set(ylabel='survive rate [%]', title=Test_env[i] + ' + ' + speed[j].capitalize())
        bplot=a.boxplot(data_tensor[i, :, j], vert=True, patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors[j]):
            patch.set_facecolor(color)


plt.setp(axs, xticks=[1,2],
         xticklabels=exp_name)
plt.tight_layout()
plt.savefig(f"./survive-ablation.png", dpi=300)
plt.show()

# %%
# Pictures
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

Test_env = ["health", "weak", "limit"]
colors_group = ["blue", "green", "orange"]

SUR_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(SUR_DIR)

name_prefix = folder_name[:3] + '_'
data_lists = []

for exp in exp_names:
    file_name = name_prefix + Test_env + '_' + exp + ".pt"
    package_name = os.path.join(PIC_DIR, folder_name, file_name)
    data_lists.append(torch.load(package_name).unsqueeze(0))

data_tensor = torch.cat(data_lists, dim=0)
data_tensor = data_tensor.view(5, 2, -1)

print(f"data_tensor: {data_tensor.shape}")
# all_data=[np.random.normal(0,std,100) for std in range(1,4)]
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))
bplot1=axes[0].boxplot(data_tensor[:, 0],
                       vert=True,
                       patch_artist=True)


bplot2 = axes[1].boxplot(data_tensor[:, 1],
                         vert=True, 
                         patch_artist=True)



#颜色填充
colors1 = ['orange', 'lightblue', 'lightgreen', 'lightyellow', 'lightcyan']
colors2 = ['pink', 'blue', 'green', 'yellow', 'cyan']
for bplot, colors in zip((bplot1, bplot2),(colors1,colors2)):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        