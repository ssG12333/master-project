import subprocess

# 要运行的 Python 文件路径
file_to_run = './predict.py'

# 使用 subprocess 模块执行外部命令，运行另一个 Python 文件
subprocess.run(['python', file_to_run, 'source=test6.mp4', 'save=True', 'show=True'])