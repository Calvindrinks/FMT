# %%
# Pictures
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

Test_env = "limit"

DIR = os.path.dirname(os.path.realpath(__file__))

folder_name = "contact"

name_prefix = folder_name[:3] + '_'
data_lists = []
exp_names = ["normal", "mixTrain_normal", "wo_reflect", "ours"]
exp_name = ["Normal", "Mix", "W/O Ref", "Ours"]
speed = ["low", "high"]
colors = ['pink', 'blue', 'green', 'yellow']

for exp in exp_names:
    file_name = name_prefix + Test_env + '_' + exp + ".pt"
    package_name = os.path.join(DIR, file_name)
    data_lists.append(torch.load(package_name).unsqueeze(0))

data_tensor = torch.cat(data_lists, dim=0)
data_tensor = data_tensor.view(4, 2, -1)

print(f"data_tensor: {torch.mean(data_tensor[:,1], dim=-1)}")
print(f"data_tensor: {torch.mean(data_tensor[:,0], dim=-1)}")
# all_data=[np.random.normal(0,std,100) for std in range(1,4)]
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8.3,3))
for i in range(2):
    a=axes[i]
    a.set(ylabel='contact time [s]', title=Test_env.capitalize() + ' + ' + speed[i].capitalize())
    bplot1=a.boxplot(data_tensor[:, i],
                        vert=True,
                        patch_artist=True)
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

#颜色填充
        
plt.setp(axes, xticks=[1,2,3,4],
         xticklabels=exp_name)
plt.savefig(f"./Agents-contact.png")

plt.show()

