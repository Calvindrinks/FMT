# %%
import numpy as np

outputonnx = np.array([-0.603742 ,2.44739, -0.312125, 0.561717, -0.994781, 2.89296, 1.8489, -0.475937, -0.806992, 3.05208, 2.21475, -0.487881])
print(outputonnx)

# %%
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
X = torch.randn((40, 8))

tsne = TSNE(n_jobs=4)
embeddings = tsne.fit_transform(X)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
plt.scatter(vis_x, vis_y, c="red", cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
# plt.show()
plt.savefig("./pictures/squares.png")

# %%
import torch
weak_fail_env_ids = torch.zeros(10, dtype=torch.bool)
weak_fail_env_ids[[2, 3, 4]] = True
a = torch.randint(0, 10, (10,))
print(a)
a[weak_fail_env_ids].copy_(-torch.ones(3).long())
print(a)

# %%
import torch

actions = torch.randperm(4*2*2)
print(actions)
actions_buf = actions.view(4, 2, 2)
print(actions_buf)
actions_buf = actions_buf.transpose(0, 1)
print(actions_buf)
actions_buf = actions_buf.view(2, 2, 2, 2)
print(actions_buf)
actions_buf = actions_buf.flatten(-2, -1)
print(actions_buf)
a = transpose(0, 1).view(2,2,2,2).flatten(-2,-1)
print(a)


# %%
import os
 
PIC_DIR = os.path.dirname(os.path.realpath(__file__))
PIC_DIR = os.path.join(PIC_DIR, "pictures")

# %%
# rewards folder
import torch
import matplotlib.pyplot as plt
import numpy as np

Test_env = "Health"
Test_env = "limit"
Test_env = "weak"


normal_survive_rates = torch.load(f"./pictures/rewards/rew_{Test_env}_normal.pt").unsqueeze(0)
mix_survive_rates = torch.load(f"./pictures/rewards/rew_{Test_env}_mixTrain_normal.pt").unsqueeze(0)
less_survive_rates = torch.load(f"./pictures/rewards/rew_{Test_env}_wo_reflect.pt").unsqueeze(0)
ours_survive_rates = torch.load(f"./pictures/rewards/rew_{Test_env}_ours.pt").unsqueeze(0)
taskpolicy_survive_rates = torch.load(f"./pictures/rewards/rew_{Test_env}_taskpolicy.pt").unsqueeze(0)
survive_rates = torch.cat((normal_survive_rates, mix_survive_rates, less_survive_rates, ours_survive_rates, taskpolicy_survive_rates), dim=0)
survive_rates = survive_rates.view(5, 2, -1)

print(f"survive_rates: {survive_rates.shape}")
# all_data=[np.random.normal(0,std,100) for std in range(1,4)]
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))
bplot1=axes[0].boxplot(survive_rates[:, 0],
                       vert=True,
                       patch_artist=True)


bplot2 = axes[1].boxplot(survive_rates[:, 1],
                         vert=True, 
                         patch_artist=True)



#颜色填充
colors1 = ['orange', 'lightblue', 'lightgreen', 'lightyellow', 'lightcyan']
colors2 = ['pink', 'blue', 'green', 'yellow', 'cyan']
for bplot, colors in zip((bplot1, bplot2),(colors1,colors2)):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
plt.savefig(f"./pictures/rewards/rew_{Test_env}.png")
plt.show()

# %%
# contact folder

import torch
import matplotlib.pyplot as plt
import numpy as np

Test_env = "limit"

contact_folder = "contact_without_filter"
normal_survive_rates = torch.load(f"./pictures/{contact_folder}/ct_{Test_env}_normal.pt").unsqueeze(0)
mix_survive_rates = torch.load(f"./pictures/{contact_folder}/ct_{Test_env}_mixTrain_normal.pt").unsqueeze(0)
less_survive_rates = torch.load(f"./pictures/{contact_folder}/ct_{Test_env}_wo_reflect.pt").unsqueeze(0)
ours_survive_rates = torch.load(f"./pictures/{contact_folder}/ct_{Test_env}_ours.pt").unsqueeze(0)
taskpolicy_survive_rates = torch.load(f"./pictures/{contact_folder}/ct_{Test_env}_taskpolicy.pt").unsqueeze(0)
survive_rates = torch.cat((normal_survive_rates, mix_survive_rates, less_survive_rates, ours_survive_rates, taskpolicy_survive_rates), dim=0)
survive_rates = survive_rates.view(5, 2, -1)

print(f"survive_rates: {survive_rates.shape}")
# all_data=[np.random.normal(0,std,100) for std in range(1,4)]
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(9,4))
bplot1=axes[0].boxplot(survive_rates[:, 0],
                       vert=True,
                       patch_artist=True)


bplot2 = axes[1].boxplot(survive_rates[:, 1],
                         vert=True, 
                         patch_artist=True)



#颜色填充
colors1 = ['orange', 'lightblue', 'lightgreen', 'lightyellow', 'lightcyan']
colors2 = ['pink', 'blue', 'green', 'yellow', 'cyan']
for bplot, colors in zip((bplot1, bplot2),(colors1,colors2)):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
plt.savefig(f"./pictures/{contact_folder}/ct_{Test_env}.png")
plt.show()

# %%
import numpy as np
a = 0
for i in range(1000):
    if 0.6 < np.random.random():
        a += 1
        print(a)

import torch
actions = torch.randn(4, 8)
commands = [0.3, 0.1, 0.05, 0.9]
commands = torch.tensor(commands)

actions *= (torch.abs(commands) > 0.1).unsqueeze(1)
print(actions)

# %%

import pyaudio
import numpy as np
import librosa

# 设置音频参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2

# 初始化PyAudio对象
audio = pyaudio.PyAudio()

# 打开音频流
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

try:
    while True:
        # 读取音频数据
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))

        # 转换为numpy数组
        audio_data = np.hstack(frames)

        # 提取节拍
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data.astype(float), sr=RATE)
        print(f"Tempo: {tempo:.2f} BPM")

        # 可以在这里添加根据节拍进行动作的代码

except KeyboardInterrupt:
    print("Recording stopped")

# 关闭和释放资源
stream.stop_stream()
stream.close()
audio.terminate()

# %%
import pyaudio
import wave
import numpy as np
import librosa
from IPython.display import Audio

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# 初始化pyaudio
p = pyaudio.PyAudio()

# 打开录音
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
np_frames = []

# 录音
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    np_frames.append(np.frombuffer(data, dtype=np.int16))

print("* done recording")

# 停止录音
stream.stop_stream()
stream.close()
p.terminate()

# 将frames数组转换为一维Numpy数组
audio_data = np.array(np_frames)

# 使用IPython.display.Audio播放录音
Audio(data=audio_data, rate=RATE)

# 保存录音到文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# %%
sudo apt install portaudio19-dev python-all-dev python3-all-dev
