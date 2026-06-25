import pygame


class Player:
    def __init__(self, maze):
        self.maze = maze
        self.cell_size = maze.cell_size
        self.start_x = maze.start_pos[0] * self.cell_size + self.cell_size // 2
        self.start_y = maze.start_pos[1] * self.cell_size + self.cell_size // 2
        self.x = self.start_x
        self.y = self.start_y
        self.speed = 5
        self.move_path = []
        self.radius = self.cell_size // 4

    def reset(self):
        self.start_x = self.maze.start_pos[0] * self.maze.cell_size + self.maze.cell_size // 2
        self.start_y = self.maze.start_pos[1] * self.maze.cell_size + self.maze.cell_size // 2
        self.x = self.start_x
        self.y = self.start_y
        self.move_path = []

    def draw(self, screen, color=(255, 255, 0)):
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        if len(self.move_path) > 1:
            pygame.draw.lines(screen, color, False, self.move_path, 2)

    def move(self, dx, dy):
        self.x += dx * self.speed
        self.y += dy * self.speed
        self.move_path.append((self.x, self.y))