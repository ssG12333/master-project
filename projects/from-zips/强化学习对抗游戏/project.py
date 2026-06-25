import pygame
import random
from pygame.locals import *

# 初始化 Pygame
pygame.init()

# 定义屏幕大小、颜色、字体等
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('N-back Task Experiment')

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 字体
font = pygame.font.SysFont('Arial', 36)
small_font = pygame.font.SysFont('Arial', 28)

# 时钟
clock = pygame.time.Clock()

# 词语库（87个词汇）
word_list = ["word" + str(i) for i in range(1, 88)]  # 用你自己的词语库替换


# 显示指导语
def display_instructions():
    screen.fill(WHITE)
    instructions = [
        "欢迎来到N-back任务实验",
        "在网格中你会看到一个词语随机出现在某个位置",
        "按J键表示当前位置与上次位置相同",
        "按F键表示当前词语与上次词语相同",
        "练习阶段：按空格键开始练习 (20轮)",
        "正式实验：完成练习后 (48轮)"
    ]
    for i, line in enumerate(instructions):
        text = font.render(line, True, BLACK)
        screen.blit(text, (50, 50 + i * 50))
    pygame.display.update()


# 显示4x4网格和随机词语
def display_grid(word, position, round_number, phase):
    screen.fill(WHITE)

    # 绘制轮次和阶段信息
    round_text = small_font.render(f"{phase} - 第{round_number}轮", True, BLACK)
    screen.blit(round_text, (10, 10))

    # 绘制4x4网格
    grid_size = 4
    cell_width = screen_width // grid_size
    cell_height = screen_height // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            rect = pygame.Rect(col * cell_width, row * cell_height, cell_width, cell_height)
            pygame.draw.rect(screen, BLACK, rect, 2)  # 网格线条

    # 在指定位置显示词语
    text = font.render(word, True, BLUE)
    x, y = position
    screen.blit(text, (x * cell_width + cell_width // 4, y * cell_height + cell_height // 4))

    pygame.display.update()


# 练习和实验阶段通用函数
def task_phase(total_trials, back_level, phase_name):
    correct_responses = 0
    words = random.sample(word_list, total_trials)
    positions = [(random.randint(0, 3), random.randint(0, 3)) for _ in range(total_trials)]

    for i in range(total_trials):
        display_grid(words[i], positions[i], i + 1, phase_name)
        response = wait_for_response()
        correct_responses += check_response(i, words, positions, response, back_level)

        pygame.time.wait(1000)  # 停留1000ms，下一轮

    accuracy = correct_responses / total_trials
    return accuracy


# 等待用户响应
def wait_for_response():
    while True:
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_j:
                    return "J"
                elif event.key == K_f:
                    return "F"


# 检查响应正确性
def check_response(index, words, positions, response, back_level):
    if index >= back_level:
        if response == "J" and positions[index] == positions[index - back_level]:
            return 1
        elif response == "F" and words[index] == words[index - back_level]:
            return 1
    return 0


# 显示正确率
def display_accuracy(accuracy, phase_name):
    screen.fill(WHITE)
    if accuracy >= 0.8:
        result_text = font.render(f"{phase_name} 完成！正确率: {accuracy * 100:.2f}%", True, GREEN)
    else:
        result_text = font.render(f"{phase_name} 结束。正确率: {accuracy * 100:.2f}% 未通过", True, RED)

    screen.blit(result_text, (screen_width // 4, screen_height // 2))
    pygame.display.update()
    pygame.time.wait(3000)  # 停留3秒


# 主函数
def main():
    display_instructions()

    # 等待用户按下空格键开始练习阶段
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_SPACE:
                waiting = False

    # 设置back任务难度
    back_level = 1  # 可以根据需要修改为1back到5back

    # 练习阶段
    accuracy = task_phase(20, back_level, "练习阶段")
    display_accuracy(accuracy, "练习阶段")

    # 如果练习通过，进入正式实验
    if accuracy >= 0.8:
        print("练习阶段成功，进入正式实验")
        accuracy = task_phase(48, back_level, "正式实验")
        display_accuracy(accuracy, "正式实验")
    else:
        print("练习阶段未通过，请重新尝试")

    pygame.quit()


if __name__ == "__main__":
    main()
