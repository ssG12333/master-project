import math
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astar_search import astar_distance
from reeds_shepp import reeds_shepp
from motion_model import find_route


SCENARIOS = {
    1: {
        "name": "Scenario 1 - Overtaking",
        "row": 3,
        "col": 10,
        "obstacles": {(0, 4): 1, (0, 5): 1},
        "start_pose": [0.5, 0.5, 0],
        "goal_pose": [9.5, 0.5, 0],
        "min_r": 0.5,
    },
    2: {
        "name": "Scenario 2 - U-turn",
        "row": 10,
        "col": 5,
        "obstacles": {(r, 2): 1 for r in range(0, 8)},
        "start_pose": [0.5, 0.5, 0],
        "goal_pose": [4.5, 0.5, 0],
        "min_r": 0.8,
    },
    3: {
        "name": "Scenario 3 - Turning Corner",
        "row": 10,
        "col": 10,
        "obstacles": {(r, c): 1 for r in range(2, 10) for c in range(2, 10)},
        "start_pose": [0.5, 9.5, 0],
        "goal_pose": [9.5, 0.5, 0],
        "min_r": 1.0,
    },
}


def mod2pi(x):
    v = x % (2 * math.pi)
    if v < -math.pi:
        v += 2 * math.pi
    elif v > math.pi:
        v -= 2 * math.pi
    return v


def plot_car_outline(ax, pose, size_car, **kwargs):
    length, width, wheelbase = size_car
    x, y, theta = pose

    alphy1 = math.atan((width / 2) / wheelbase)
    alphy2 = -alphy1
    alphy3 = math.pi + math.atan((width / 2) / (length - wheelbase))
    alphy4 = math.pi - math.atan((width / 2) / (length - wheelbase))

    angles = [alphy1, alphy2, alphy3, alphy4, alphy1]
    angles = [a + theta for a in angles]

    l1 = math.sqrt((width / 2) ** 2 + wheelbase ** 2)
    l2 = math.sqrt((width / 2) ** 2 + (length - wheelbase) ** 2)
    lengths = [l1, l1, l2, l2, l1]

    cx = [x + l * math.cos(a) for l, a in zip(lengths, angles)]
    cy = [y + l * math.sin(a) for l, a in zip(lengths, angles)]

    defaults = {'color': 'k', 'linewidth': 1, 'alpha': 0.6}
    defaults.update(kwargs)
    ax.plot(cx, cy, **defaults)


def draw_obstacles(ax, sign, row, col):
    ob_coo = []
    for i in range(row):
        for j in range(col):
            if sign[i][j] == 1:
                y = [i, i, i + 1, i + 1]
                x = [j, j + 1, j + 1, j]
                ax.fill(x, y, 'k', alpha=1)
                ob_coo.append([j + 0.5, i + 0.5])
    return np.array(ob_coo) if len(ob_coo) > 0 else np.empty((0, 2))


def draw_grid(ax, row, col):
    for i in range(1, row + 1):
        ax.plot([0, col], [i, i], 'k-', linewidth=0.5)
    for i in range(1, col + 1):
        ax.plot([i, i], [0, row], 'k-', linewidth=0.5)


class Node:
    def __init__(self, x, y, theta, direction, g, h, f, node_index, parent_index):
        self.x = x
        self.y = y
        self.theta = theta
        self.direction = direction
        self.g = g
        self.h = h
        self.f = f
        self.node_index = node_index
        self.parent_index = parent_index
        self.route = []


def build_obstacle_grid(row, col, obstacles):
    sign = np.zeros((row, col))
    for (r, c), val in obstacles.items():
        if 0 <= r < row and 0 <= c < col:
            sign[r][c] = val
    return sign


def hybrid_astar_search(start_pose, goal_pose, sign, row, col,
                         min_r, safe_dis, step, P3, ax, show_tree=False):
    start_grid = [int(math.ceil(start_pose[1])), int(math.ceil(start_pose[0]))]
    goal_grid = [int(math.ceil(goal_pose[1])), int(math.ceil(goal_pose[0]))]

    t_start = time.time()

    dis_astar = astar_distance(start_grid, goal_grid, sign)
    dis_rs, _ = reeds_shepp(start_pose, goal_pose, min_r)
    if np.isinf(dis_rs):
        dis_rs = dis_astar + 100
    h_value = max(dis_astar, dis_rs)

    opened = []
    node_array = []
    visited = {}

    start_node = Node(
        x=start_pose[0], y=start_pose[1], theta=start_pose[2],
        direction=0, g=0, h=h_value, f=h_value,
        node_index=1, parent_index=0
    )
    start_node.route = [start_pose]
    opened.append(start_node)
    node_array.append(start_node)

    node_index = 2
    max_iterations = 80000
    iteration = 0
    now_point = start_node
    nodes_expanded = 0
    nodes_generated = 0

    while True:
        if (int(math.ceil(now_point.x)) == int(math.ceil(goal_pose[0])) and
            int(math.ceil(now_point.y)) == int(math.ceil(goal_pose[1]))):
            break

        min_f = float('inf')
        best_idx = -1
        for i, node in enumerate(opened):
            if node.f < min_f:
                min_f = node.f
                best_idx = i

        if best_idx == -1:
            return None, {'time': time.time() - t_start, 'expanded': nodes_expanded, 'generated': nodes_generated}

        now_point = opened[best_idx]
        opened.pop(best_idx)
        nodes_expanded += 1

        if np.linalg.norm([now_point.x - goal_pose[0],
                          now_point.y - goal_pose[1]]) < 0.5 * step:
            break

        iteration += 1
        current_pose = [now_point.x, now_point.y, now_point.theta]

        for i in range(1, 7):
            isok, x, y, sita, route = find_route(
                current_pose, i, step, min_r, safe_dis, ob_coo)

            if x <= 0 or x > col or y <= 0 or y > row:
                isok = 1

            if isok == 0:
                state_key = (round(x * 4) / 4, round(y * 4) / 4, round(sita * 4) / 4)
                if state_key in visited:
                    continue
                visited[state_key] = True
                nodes_generated += 1

                direction = 0 if i <= 3 else 1
                g_cost = now_point.g + step + abs(sita - now_point.theta) * P3

                temp_grid = [int(math.ceil(y)), int(math.ceil(x))]
                dis_astar = astar_distance(temp_grid, goal_grid, sign)
                dis_rs, _ = reeds_shepp([x, y, sita], goal_pose, min_r)

                if np.isinf(dis_rs):
                    dis_rs = dis_astar + 100

                h_cost = max(dis_astar, dis_rs)
                f_cost = g_cost + h_cost

                new_node = Node(
                    x=x, y=y, theta=sita, direction=direction,
                    g=g_cost, h=h_cost, f=f_cost,
                    node_index=node_index, parent_index=now_point.node_index
                )
                new_node.route = route.tolist()

                opened.append(new_node)
                node_array.append(new_node)
                node_index += 1

                if show_tree:
                    if i <= 3:
                        ax.plot(route[:, 0], route[:, 1], 'b-',
                               linewidth=1.0, alpha=0.35)
                    else:
                        ax.plot(route[:, 0], route[:, 1], 'g-',
                               linewidth=1.0, alpha=0.35)
                    if iteration % 10 == 0:
                        ax.plot(route[-1, 0], route[-1, 1], 'o',
                               markersize=3, markerfacecolor='r',
                               markeredgecolor='none', alpha=0.5)

        if iteration >= max_iterations:
            return None, {'time': time.time() - t_start, 'expanded': nodes_expanded, 'generated': nodes_generated}

    node_temp = now_point
    index = [now_point.node_index]

    while node_temp.parent_index != 0:
        found = False
        for node in node_array:
            if node.node_index == node_temp.parent_index:
                node_temp = node
                index.append(node_temp.node_index)
                found = True
                break
        if not found:
            break

    index.reverse()

    route_all = [start_pose]
    for idx in index:
        for node in node_array:
            if node.node_index == idx:
                route = np.array(node.route)
                route_all.extend(route[1:].tolist())
                break

    route_all = np.array(route_all)

    _, final_route = reeds_shepp(route_all[-1, :3], goal_pose, min_r)
    if len(final_route) > 1:
        final_route = np.array(final_route)
        route_all = np.vstack([route_all, final_route[1:]])
    else:
        last_pose = route_all[-1, :3]
        dx = goal_pose[0] - last_pose[0]
        dy = goal_pose[1] - last_pose[1]
        if np.linalg.norm([dx, dy]) > 0.1:
            n_pts = max(5, int(np.linalg.norm([dx, dy]) / step * 10))
            xs = np.linspace(last_pose[0], goal_pose[0], n_pts)
            ys = np.linspace(last_pose[1], goal_pose[1], n_pts)
            angles = np.full(n_pts, goal_pose[2])
            extra = np.column_stack([xs, ys, angles])
            route_all = np.vstack([route_all, extra[1:]])

    stats = {
        'time': time.time() - t_start,
        'expanded': nodes_expanded,
        'generated': nodes_generated,
        'path_length': len(route_all),
    }
    return route_all, stats


def save_vehicle_path(scenario_id, route_all, sign, row, col, cfg):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, col)
    ax.set_ylim(0, row)
    ax.set_xlabel('X / m', fontsize=12)
    ax.set_ylabel('Y / m', fontsize=12)

    for i in range(row):
        for j in range(col):
            if sign[i][j] == 1:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='k')
                ax.add_patch(rect)

    size_car = [1.0, 0.6, 0.4]
    step_pose = max(1, len(route_all) // 15)
    for i in range(0, len(route_all), step_pose):
        plot_car_outline(ax, route_all[i], size_car, color='k', linewidth=1.2, alpha=0.7)

    ax.plot(route_all[:, 0], route_all[:, 1], 'r-', linewidth=1.5, alpha=0.5)

    dis_all = sum(np.linalg.norm(route_all[i, 0:2] - route_all[i - 1, 0:2]) for i in range(1, len(route_all)))
    ax.set_title(f'{cfg["name"]} (Total: {dis_all:.2f}m)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filename = f"scenario_{scenario_id}_vehicle_path.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_rear_axle_center(scenario_id, route_all, sign, row, col, cfg):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, col)
    ax.set_ylim(0, row)
    ax.set_xlabel('X / m', fontsize=12)
    ax.set_ylabel('Y / m', fontsize=12)

    for i in range(row):
        for j in range(col):
            if sign[i][j] == 1:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='k')
                ax.add_patch(rect)

    ax.plot(route_all[:, 0], route_all[:, 1], 'r-', linewidth=2)

    dis_all = sum(np.linalg.norm(route_all[i, 0:2] - route_all[i - 1, 0:2]) for i in range(1, len(route_all)))
    ax.set_title(f'Rear Axle Center Trajectory (Total: {dis_all:.2f}m)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    filename = f"scenario_{scenario_id}_rear_axle.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_steering_angle(scenario_id, route_all, cfg):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel('X / m', fontsize=12)
    ax.set_ylabel('Steering Angle / rad', fontsize=12)

    steering_angles = []
    for i in range(1, len(route_all)):
        dx = route_all[i, 0] - route_all[i - 1, 0]
        dy = route_all[i, 1] - route_all[i - 1, 1]
        if np.linalg.norm([dx, dy]) > 0.01:
            actual_heading = math.atan2(dy, dx)
            delta = actual_heading - route_all[i - 1, 2]
            delta = mod2pi(delta)
            steering_angles.append(abs(delta))
        else:
            steering_angles.append(0)

    if len(steering_angles) > 1:
        x_pts = min(len(steering_angles), len(route_all) - 1)
        ax.plot(route_all[1:x_pts + 1, 0], steering_angles[:x_pts], 'b-', linewidth=1.5)
        max_steer = max(steering_angles) if steering_angles else 1
        ax.set_ylim(0, max_steer * 1.3 if max_steer > 0 else 1)

    ax.set_title('Steering Angle Profile', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f"scenario_{scenario_id}_steering.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_curvature_profile(scenario_id, route_all, cfg):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel('X / m', fontsize=12)
    ax.set_ylabel('Curvature / (1/m)', fontsize=12)

    curvatures = []
    for i in range(1, len(route_all)):
        dx = route_all[i, 0] - route_all[i - 1, 0]
        dy = route_all[i, 1] - route_all[i - 1, 1]
        if np.linalg.norm([dx, dy]) > 0.01:
            heading = math.atan2(dy, dx)
            curv = heading - route_all[i - 1, 2]
            curv = mod2pi(curv)
            seg_len = np.linalg.norm([dx, dy])
            curvatures.append(abs(curv) / seg_len if seg_len > 0.001 else 0)
        else:
            curvatures.append(0)

    if len(curvatures) > 1:
        x_pts = min(len(curvatures), len(route_all) - 1)
        ax.plot(route_all[1:x_pts + 1, 0], curvatures[:x_pts], 'm-', linewidth=1.5)
        ax.set_ylim(0, max(curvatures) * 1.3 if max(curvatures) > 0 else 1)

    ax.set_title('Curvature Profile', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f"scenario_{scenario_id}_curvature.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_search_tree(scenario_id, route_all, sign, row, col, start_pose, goal_pose):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, col)
    ax.set_ylim(0, row)

    for i in range(row):
        for j in range(col):
            if sign[i][j] == 1:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='k')
                ax.add_patch(rect)

    ax.plot(start_pose[0], start_pose[1], 'p', markersize=10,
            markerfacecolor='b', markeredgecolor='m', label='Start')
    ax.plot(goal_pose[0], goal_pose[1], 'o', markersize=10,
            markerfacecolor='g', markeredgecolor='c', label='Goal')

    ax.plot(route_all[:, 0], route_all[:, 1], 'r-', linewidth=2, label='Path')
    ax.legend(loc='upper right')

    cfg = SCENARIOS[scenario_id]
    ax.set_title(f'{cfg["name"]} - Search Result', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f"scenario_{scenario_id}_tree.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_pose_analysis(scenario_id, route_all, cfg):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    x_vals = route_all[:, 0]
    y_vals = route_all[:, 1]
    theta_vals = route_all[:, 2]
    
    axes[0].plot(x_vals, y_vals, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Y / m', fontsize=11)
    axes[0].set_title('Position Trajectory', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_vals, theta_vals, 'r-', linewidth=1.5)
    axes[1].set_ylabel('Heading / rad', fontsize=11)
    axes[1].set_title('Heading Angle Profile', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x_vals[:-1], np.diff(theta_vals), 'g-', linewidth=1.5)
    axes[2].set_xlabel('X / m', fontsize=11)
    axes[2].set_ylabel('d(heading)/dx / (rad/m)', fontsize=11)
    axes[2].set_title('Heading Change Rate', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'{cfg["name"]} - Pose Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    filename = f"scenario_{scenario_id}_pose_analysis.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_obstacle_clearance(scenario_id, route_all, sign, row, col, cfg):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel('Path Index', fontsize=11)
    ax.set_ylabel('Clearance / m', fontsize=11)
    
    clearances = []
    for pose in route_all:
        min_dist = float('inf')
        for i in range(row):
            for j in range(col):
                if sign[i][j] == 1:
                    obs_x, obs_y = j + 0.5, i + 0.5
                    dist = np.linalg.norm([pose[0] - obs_x, pose[1] - obs_y])
                    min_dist = min(min_dist, dist)
        clearances.append(min_dist)
    
    ax.plot(range(len(clearances)), clearances, 'm-', linewidth=1.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Safety Margin')
    ax.legend()
    ax.set_title('Obstacle Clearance Along Path', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"scenario_{scenario_id}_clearance.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    return min(clearances) if clearances else 0


def save_path_segment_analysis(scenario_id, route_all, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    segment_types = []
    segment_lengths = []
    
    for i in range(1, len(route_all)):
        dx = route_all[i, 0] - route_all[i - 1, 0]
        dy = route_all[i, 1] - route_all[i - 1, 1]
        dtheta = route_all[i, 2] - route_all[i - 1, 2]
        seg_len = np.linalg.norm([dx, dy])
        
        if abs(dtheta) < 0.05:
            segment_types.append('Straight')
        else:
            segment_types.append('Curve')
        segment_lengths.append(seg_len)
    
    cumulative = np.cumsum([0] + segment_lengths)
    x_vals = cumulative[1:]
    
    straight_mask = [t == 'Straight' for t in segment_types]
    curve_mask = [t == 'Curve' for t in segment_types]
    
    axes[0].bar(x_vals, segment_lengths, color='skyblue', label='Straight', alpha=0.7)
    axes[0].bar(x_vals, np.where(curve_mask, segment_lengths, 0), color='coral', label='Curve', alpha=0.7)
    axes[0].set_ylabel('Segment Length / m', fontsize=11)
    axes[0].set_title('Path Segment Composition', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    n_straight = sum(straight_mask)
    n_curve = sum(curve_mask)
    axes[1].pie([n_straight, n_curve], labels=['Straight', 'Curve'], autopct='%1.1f%%',
                colors=['skyblue', 'coral'])
    axes[1].set_title(f'Segment Distribution (Straight: {n_straight}, Curve: {n_curve})', fontsize=12)
    
    plt.tight_layout()
    filename = f"scenario_{scenario_id}_segments.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_comparison_table(results):
    if len(results) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    
    headers = ['Scenario', 'Distance (m)', 'Nodes Expanded', 'Nodes Generated', 
               'Path Points', 'Time (s)', 'Min Clearance (m)', 'Efficiency (pts/s)']
    
    table_data = []
    for r in results:
        efficiency = r['path_length'] / r['time'] if r['time'] > 0 else 0
        table_data.append([
            r['name'],
            f"{r['distance']:.2f}",
            str(r['expanded']),
            str(r['generated']),
            str(r['path_length']),
            f"{r['time']:.3f}",
            f"{r.get('min_clearance', 0):.3f}",
            f"{efficiency:.1f}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(results)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Hybrid A* Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = "performance_comparison_table.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_csv_table(results):
    import csv
    
    if not results:
        return
    
    filename = "performance_data.csv"
    
    headers = ['Scenario', 'Distance (m)', 'Nodes Expanded', 'Nodes Generated', 
               'Path Points', 'Time (s)', 'Min Clearance (m)', 'Efficiency (pts/s)']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for r in results:
            efficiency = r['path_length'] / r['time'] if r['time'] > 0 else 0
            writer.writerow([
                r['name'],
                f"{r['distance']:.2f}",
                r['expanded'],
                r['generated'],
                r['path_length'],
                f"{r['time']:.3f}",
                f"{r.get('min_clearance', 0):.3f}",
                f"{efficiency:.1f}"
            ])
    
    print(f"  Saved CSV: {filename}")


def save_scenario_comparison_plot(results, route_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for idx, (r, rd) in enumerate(zip(results, route_data)):
        route_all, sign, row, col = rd
        cfg = SCENARIOS[idx + 1]
        
        ax.plot(route_all[:, 0], route_all[:, 1], color=colors[idx], 
               linewidth=2, label=f'{cfg["name"]} ({r["distance"]:.2f}m)',
               marker=markers[idx], markersize=3, markevery=20)
    
    ax.set_xlabel('X / m', fontsize=12)
    ax.set_ylabel('Y / m', fontsize=12)
    ax.set_title('Path Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    filename = "path_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def save_all_analysis(scenario_id, route_all, sign, row, col, stats):
    cfg = SCENARIOS[scenario_id]
    start_pose = np.array(cfg['start_pose'])
    goal_pose = np.array(cfg['goal_pose'])

    save_vehicle_path(scenario_id, route_all, sign, row, col, cfg)
    min_clearance = save_obstacle_clearance(scenario_id, route_all, sign, row, col, cfg)
    
    return {'route_all': route_all, 'sign': sign, 'row': row, 'col': col, 'min_clearance': min_clearance}


def run_scenario(scenario_id):
    cfg = SCENARIOS[scenario_id]
    scene_name = cfg["name"]

    row = cfg["row"]
    col = cfg["col"]
    sign = build_obstacle_grid(row, col, cfg["obstacles"])
    start_pose = np.array(cfg["start_pose"])
    goal_pose = np.array(cfg["goal_pose"])
    min_r = cfg["min_r"]
    safe_dis = 0.5
    step = 0.5
    P3 = 0.01

    print(f"\n{'='*50}")
    print(f"  {scene_name}")
    print(f"{'='*50}")

    global ob_coo

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_xlim(0, col)
    ax.set_ylim(0, row)

    ob_coo = draw_obstacles(ax, sign, row, col)
    draw_grid(ax, row, col)

    ax.plot(start_pose[0], start_pose[1], 'p', markersize=10,
            markerfacecolor='b', markeredgecolor='m', label='Start')
    ax.plot(goal_pose[0], goal_pose[1], 'o', markersize=10,
            markerfacecolor='g', markeredgecolor='c', label='Goal')

    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(scene_name, fontsize=14)

    route_all, stats = hybrid_astar_search(
        start_pose, goal_pose, sign, row, col,
        min_r, safe_dis, step, P3, ax, show_tree=True
    )

    if route_all is None:
        print("  Path planning failed!")
        plt.savefig(f"scenario_{scenario_id}_failed.png", dpi=150, bbox_inches='tight')
        plt.close()
        return None

    dis_all = sum(np.linalg.norm(route_all[i, 0:2] - route_all[i - 1, 0:2])
                  for i in range(1, len(route_all)))
    print(f"  Path length: {dis_all:.2f}m")
    print(f"  Nodes expanded: {stats['expanded']}")
    print(f"  Nodes generated: {stats['generated']}")
    print(f"  Computation time: {stats['time']:.3f}s")
    print(f"  Path points: {stats['path_length']}")

    fig.savefig(f"scenario_{scenario_id}_tree.png", dpi=150, bbox_inches='tight')
    print(f"  Saved search tree: scenario_{scenario_id}_tree.png")
    plt.close()
    
    analysis_data = save_all_analysis(scenario_id, route_all, sign, row, col, stats)

    return {'name': scene_name, 'distance': dis_all, **stats, **analysis_data}


if __name__ == '__main__':
    print("Hybrid A* Path Planning - Analysis")
    print("=" * 50)

    scenarios_to_run = [1, 2, 3]
    results = []
    route_data = []

    for sid in scenarios_to_run:
        result = run_scenario(sid)
        if result:
            results.append(result)
            route_data.append((result['route_all'], result['sign'], result['row'], result['col']))

    if results:
        print("\n" + "=" * 80)
        print("  Performance Comparison")
        print("=" * 80)
        print(f"{'Scenario':<25} {'Dist(m)':<10} {'Expanded':<10} {'Generated':<10} {'Time(s)':<10} {'Clearance':<12} {'Efficiency':<10}")
        print("-" * 80)
        for r in results:
            eff = r['path_length'] / r['time'] if r['time'] > 0 else 0
            print(f"{r['name']:<25} {r['distance']:<10.2f} {r['expanded']:<10} {r['generated']:<10} {r['time']:<10.3f} {r.get('min_clearance', 0):<12.3f} {eff:<10.1f}")

        save_comparison_table(results)
        save_csv_table(results)
        save_scenario_comparison_plot(results, route_data)
