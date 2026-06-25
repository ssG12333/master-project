from enum import Enum
# state.py中统一定义（删除main代码中的重复定义）
class GameState(Enum):
        MENU = 0
        PLAYING = 1
        PAUSED = 2
        SUCCESS = 3
        PERFORMANCE_ANALYSIS = 4  # 新增