# %%
import pygame
import sys

# 初始化 Pygame
pygame.init()

# 初始化手柄
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
num = joystick.get_numaxes()
print(num)
joystick.init()

# 设置窗口大小
width, height = 400, 300

# 创建窗口
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Pygame Joystick Event')

# 设置字体和字体大小
font = pygame.font.Font(None, 36)

def in_deadzone(value, center, threshold):
    return abs(value - center) < threshold

def main():
    while True:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()
        pygame.event.get()

        # 获取左摇杆的值
        left_stick_x = joystick.get_axis(3)
        left_stick_y = joystick.get_axis(1)
        L1 = joystick.get_button(4)
        R1 = joystick.get_button(5) 
        right_key = pygame.key.get_pressed()[pygame.K_RIGHT]
        
        if in_deadzone(left_stick_x, 0, 0.1):
            left_stick_x = 0
        if in_deadzone(left_stick_y, 0, 0.1):
            left_stick_y = 0


        # 清除屏幕
        screen.fill((255, 255, 255))

        # 在屏幕上渲染左摇杆的值
        text = font.render(f'Left Stick: X={left_stick_x:.2f}, Y={left_stick_y:.2f}', True, (0, 0, 0))
        textbutton = font.render(f'L1: {L1}, R1: {R1}', True, (255, 0, 0))
        textR1 = font.render(f"R1: {R1}", True, (25, 255, 0))
        screen.blit(text, (50, 80))
        screen.blit(textbutton, (50, 140))
        
        if right_key == 1 and prev_right_key == 0:
            print("右方向键被按下（下降沿）")
        if R1 == 1 and prev_r1_button == 0:
            print("R1 按键被按下（下降沿）")
            pygame.quit()
            sys.exit()
        screen.blit(textR1, (50, 180))
        
        prev_r1_button = R1
        prev_right_key = right_key
        pygame.display.flip()

if __name__ == "__main__":
    main()


# %%
import pygame
import sys

# 初始化 Pygame
pygame.init()

# 设置窗口大小
width, height = 400, 300

# 创建窗口
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Pygame Joystick Event')

# 设置字体和字体大小
font = pygame.font.Font(None, 36)

def in_deadzone(value, center, threshold):
    return abs(value - center) < threshold

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.event.get()

        right_key = pygame.key.get_pressed()[pygame.K_RIGHT]
        if right_key == 1 and prev_right_key == 0:
            print("右方向键被按下（下降沿）")
        prev_right_key = right_key

        # 清除屏幕
        screen.fill((255, 255, 255))

        # 在屏幕上渲染左摇杆的值
        textbutton = font.render(f"R1: {right_key}", True, (25, 255, 0))
        screen.blit(textbutton, (50, 140))
        
        
        prev_right_key = right_key
        pygame.display.flip()

if __name__ == "__main__":
    main()

# %%
import torch
import torch.nn as nn

# Step 1: 判断observations[..., 4]是否大于0，得到布尔张量
observations = torch.randn(2, 3, 6)
bool_tensor = torch.gt(observations[..., 4], 0)

# Step 2: 将布尔张量转换为整数张量
int_tensor = bool_tensor.long()

# Step 3: 创建一个嵌入层，输入维度为2，输出维度为10
embedding_layer = nn.Embedding(2, 10)

# Step 4: 使用嵌入层将整数张量嵌入到10维向量中
embedded_tensor = embedding_layer(int_tensor)

print(observations)
print(bool_tensor)
print(int_tensor)
print(embedded_tensor)

# %%
help(torch.gt)