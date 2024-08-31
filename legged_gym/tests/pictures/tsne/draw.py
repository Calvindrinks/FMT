# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(n_jobs=4)

fault_group_num = 4
num_envs = 24
fault_group = torch.div(torch.arange(num_envs), (num_envs/fault_group_num), rounding_mode='floor').to(torch.long)

env_group_name = ["health", "weak", "limit"]
colors_group = ["lightblue", "pink", "orange"]

name_list = ["tsne-faultLeg-actions", "tsne-faultLeg-states", "tsne-states"]
xlabel_list = ['Hip Joints Actions [rad]', 'Hip Joints Positions [rad]', 'Robot Joints Positions t-SNE']
ylabel_list = ['Knee Joints Actions [rad]', 'Knee Joints Positions [rad]', 'Robot Joints Positions t-SNE']
title_list = ["Fault Leg Actions", "Fault Leg States", "t-SNE of Robot States"]


nb_rows, nb_cols = (1, 3)
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(15, 5))
for j in range(len(name_list)):
    
    a = axs[j]
    name = name_list[j]
    # a.title(name)
    # a.set(xlabel=xlabel_list[j], ylabel=ylabel_list[j])
    a.set_xlabel(xlabel_list[j], fontsize=16)  # 设置 x 轴标签
    a.set_ylabel(ylabel_list[j], fontsize=16)  # 设置 y 轴标签
    a.set_title(title_list[j], fontsize=20)

    buffer = torch.load(f"./{name}.pt")
    for i in range(1, fault_group_num):
        actions_group = buffer[fault_group==i]
        if j>=2:
            embeddings = tsne.fit_transform(actions_group.flatten(0, 1).flatten(-2, -1))
        else:
            embeddings = actions_group.flatten(0, 1).flatten(-2, -1)
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        a.scatter(vis_x, vis_y, c=colors_group[i-1], marker='.', label=env_group_name[i-1].capitalize())
        a.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig("./font/act-state-6Agents-font.png", dpi=300)
plt.show()
# %%
import cv2

# 读取整幅大图
large_image = cv2.imread('./font/act-state-6Agents-font.png')

# 获取大图的宽度和高度
height, width = large_image.shape[:2]

# 分割成三张小图
small_image_a = large_image[:, :width//3]
small_image_b = large_image[:, width//3:2*width//3]
small_image_c = large_image[:, 2*width//3:]

# 保存小图
cv2.imwrite('./font/leg-action.png', small_image_a)
cv2.imwrite('./font/leg-states.png', small_image_b)
cv2.imwrite('./font/robot-states-tsne.png', small_image_c)

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE

tsne = TSNE(n_jobs=4)

fault_group_num = 4
num_envs = 24
fault_group = torch.div(torch.arange(num_envs), (num_envs/fault_group_num), rounding_mode='floor').to(torch.long)
colors_group = ["lightblue", "lightblue", "pink", "orange"]

fault_leg = 2
if fault_leg:
    name_list = ["tsne-actions", "tsne-states"] 
else:
    name_list = ["tsne-faultLeg-actions", "tsne-faultLeg-states"]

for j in range(len(name_list)):

    name = name_list[j]
    plt.subplot(2, 3, j+1)
    plt.title(name)

    buffer = torch.load(f"./{name}.pt")
    for i in range(fault_group_num):
        actions_group = buffer[fault_group==i]
        if fault_leg==1:
            embeddings = actions_group.flatten(0, 1).flatten(-2, -1)
        else:
            embeddings = tsne.fit_transform(actions_group.flatten(0, 1).flatten(-2, -1))
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        plt.scatter(vis_x, vis_y, c=colors_group[i], cmap=plt.cm.get_cmap("jet", 10), marker='.')
        # plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
plt.show()
