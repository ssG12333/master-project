import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_car(pose, size_car):
    length, width, wheelbase = size_car
    x, y, theta = pose
    
    car_points = np.array([
        [x + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
         y + length / 2 * np.sin(theta) + width / 2 * np.cos(theta)],
        [x + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
         y + length / 2 * np.sin(theta) - width / 2 * np.cos(theta)],
        [x - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
         y - length / 2 * np.sin(theta) - width / 2 * np.cos(theta)],
        [x - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
         y - length / 2 * np.sin(theta) + width / 2 * np.cos(theta)]
    ])
    
    car = patches.Polygon(car_points, closed=True, edgecolor='r', facecolor='lightblue', linewidth=2)
    
    front_x = x + (length / 2) * np.cos(theta)
    front_y = y + (length / 2) * np.sin(theta)
    front = patches.Circle((front_x, front_y), radius=0.1, color='red')
    
    return car, front

def draw_obstacles(sign, row, col):
    for i in range(row):
        for j in range(col):
            if sign[i][j] == 1:
                y = [i - 1, i - 1, i, i]
                x = [j - 1, j, j, j - 1]
                h = plt.fill(x, y, 'k', alpha=1)

def draw_grid(row, col):
    for i in range(row + 1):
        plt.plot([0, col], [i, i], 'k-', linewidth=0.5)
    for i in range(col + 1):
        plt.plot([i, i], [0, row], 'k-', linewidth=0.5)

def plot_path(route_all, size_car, title="Hybrid A* Path Planning"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    for i in range(1, len(route_all)):
        ax.plot([route_all[i-1][0], route_all[i][0]], 
                [route_all[i-1][1], route_all[i][1]], 
                'r-', linewidth=2)
    
    car_pose = route_all[0]
    car, front = plot_car(car_pose, size_car)
    ax.add_patch(car)
    ax.add_patch(front)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def animate_path(route_all, size_car, sign, row, col, title="Hybrid A* Path Planning"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0, col)
    ax.set_ylim(0, row)
    
    draw_obstacles(sign, row, col)
    draw_grid(row, col)
    
    ax.plot(route_all[0][0], route_all[0][1], 'bp', markersize=10, 
            markerfacecolor='b', markeredgecolor='m')
    ax.plot(route_all[-1][0], route_all[-1][1], 'go', markersize=10, 
            markerfacecolor='g', markeredgecolor='c')
    
    car, front = plot_car(route_all[0], size_car)
    ax.add_patch(car)
    ax.add_patch(front)
    
    path_line, = ax.plot([], [], 'r-', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    for i in range(1, len(route_all)):
        path_line.set_data([route_all[0:i, 0], route_all[0:i, 1]])
        
        car.remove()
        front.remove()
        car, front = plot_car(route_all[i], size_car)
        ax.add_patch(car)
        ax.add_patch(front)
        
        ax.plot([route_all[i-1][0], route_all[i][0]], 
                [route_all[i-1][1], route_all[i][1]], 
                'r-', linewidth=2)
        
        plt.pause(0.01)
    
    plt.show()
