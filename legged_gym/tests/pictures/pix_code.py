# %%
# Pictures
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

Test_env = "health"
Test_env = "weak"
Test_env = "limit"

PIC_DIR = os.path.dirname(os.path.realpath(__file__))
prefix = input("pictures folder(you can use prefix of folder name)")
for folder in os.listdir():
    if folder.startswith(prefix):
        folder_name = folder
        print(f"Enter folder: {folder_name}")
        break
else:
    folder_name = "survive"

name_prefix = folder_name[:3] + '_'
data_lists = []
exp_names = ["normal", "mixTrain_normal", "wo_reflect", "ours", "taskpolicy"]

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
        
plt.savefig(f"./{folder_name}/{name_prefix}{Test_env}.png")
plt.show()

# %%
### Rename files

import os
import time
from datetime import datetime

MarkWithDates = False

PIC_DIR = os.path.dirname(os.path.realpath(__file__))
folder_name = "contact"

def rename_files(directory, old_prefix, new_prefix, suffix=""):
    for filename in os.listdir(directory):
        if filename.endswith(suffix) and filename.startswith(old_prefix):
            new_name = filename.replace(old_prefix, new_prefix, 1)
            print(new_name)
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

# Usage example
old_prefix = 'ct_'
new_prefix = 'con_'

if MarkWithDates:
    dates_prefix = datetime.now().strftime('%b%d_%H-%M-%S') + '-'
    new_prefix = dates_prefix
    rename_files(folder_name, old_prefix, new_prefix, ".png")
else:
    rename_files(folder_name, old_prefix, new_prefix)