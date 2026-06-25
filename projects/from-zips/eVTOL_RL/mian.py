# main.py
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端避免tkinter警告
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import warnings
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.obstacles import ObstacleGenerator, TrainingStage

# 忽略字体相关警告
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
warnings.filterwarnings("ignore", message=".*iCCP.*")

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def visualize_obstacles(generator: ObstacleGenerator, title: str, ax, config=None):
    """在指定 Ax 上可视化障碍物 - 增强版"""
    area_x, area_y = generator.area_size
    ax.set_xlim(0, area_x)
    ax.set_ylim(0, area_y)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)

    # 如果有障碍物，根据高度着色
    if generator.obstacles:
        max_height = generator.params.get('height_max', 100)

        # 可视化障碍物
        for obs in generator.obstacles:
            # 根据高度着色（从蓝色到红色）
            height_ratio = obs['height'] / max_height
            color = plt.cm.viridis(height_ratio)

            circle = Circle(obs['position'][:2], obs['radius'],
                            facecolor=color, edgecolor='black',
                            alpha=0.7, linewidth=0.5)
            ax.add_patch(circle)

            # 中心点
            ax.plot(obs['position'][0], obs['position'][1], 'k.', markersize=1)

        # 添加颜色条（仅在有障碍物时）
        if len(generator.obstacles) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(vmin=0, vmax=max_height))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('高度 (m)', rotation=270, labelpad=15, fontsize=9)

    # 显示详细统计信息
    stats = generator.get_obstacle_statistics()
    info_text = f"数量: {stats['count']}\n"
    info_text += f"密度: {stats['density_actual']:.2e}/m²\n"

    if stats['count'] > 0:
        info_text += f"平均高度: {stats['avg_height']:.1f}m\n"
        info_text += f"平均半径: {stats['avg_radius']:.1f}m\n"

        if stats['min_gap_width'] is not None:
            info_text += f"最小间隙: {stats['min_gap_width']:.1f}m\n"
            info_text += f"安全比率: {stats['wingspan_safety_ratio']:.2f}"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))


def visualize_obstacles_with_danger(generator: ObstacleGenerator, title: str, ax, config=None):
    """带危险区域标注的可视化"""
    area_x, area_y = generator.area_size
    ax.set_xlim(0, area_x)
    ax.set_ylim(0, area_y)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)

    # 统计危险区域
    safe_count = 0
    caution_count = 0
    danger_count = 0

    # 先绘制危险区域连线（采样显示，避免过于密集）
    if len(generator.obstacles) >= 2:
        sample_rate = max(0.01, min(0.1, 100.0 / len(generator.obstacles)))  # 自适应采样率

        for i in range(len(generator.obstacles)):
            for j in range(i + 1, len(generator.obstacles)):
                obs1, obs2 = generator.obstacles[i], generator.obstacles[j]
                center_dist = np.linalg.norm(obs1['position'][:2] - obs2['position'][:2])
                gap = center_dist - obs1['radius'] - obs2['radius']

                # 分类间隙
                min_physical = generator.params.get('min_physical_gap', 0)
                min_safe = generator.params.get('min_gap_width', float('inf'))

                if gap < min_physical:
                    color = 'red'
                    alpha = 0.4
                    linewidth = 1.5
                    danger_count += 1
                    draw = True
                elif gap < min_safe:
                    color = 'orange'
                    alpha = 0.25
                    linewidth = 1.0
                    caution_count += 1
                    draw = np.random.random() < sample_rate
                else:
                    safe_count += 1
                    draw = False

                # 画连线
                if draw:
                    ax.plot([obs1['position'][0], obs2['position'][0]],
                            [obs1['position'][1], obs2['position'][1]],
                            color=color, linewidth=linewidth, alpha=alpha, zorder=1)

    # 绘制障碍物
    if generator.obstacles:
        max_height = generator.params.get('height_max', 100)

        for obs in generator.obstacles:
            height_ratio = obs['height'] / max_height
            color = plt.cm.viridis(height_ratio)

            circle = Circle(obs['position'][:2], obs['radius'],
                            facecolor=color, edgecolor='black',
                            alpha=0.7, linewidth=0.8, zorder=2)
            ax.add_patch(circle)

    # 统计信息（包含危险区域）
    stats = generator.get_obstacle_statistics()
    total_gaps = safe_count + caution_count + danger_count

    info_text = f"数量: {stats['count']}\n"
    if stats['min_gap_width'] is not None:
        info_text += f"最小间隙: {stats['min_gap_width']:.1f}m\n"

    if total_gaps > 0:
        info_text += f"\n间隙分类:\n"
        info_text += f"安全: {safe_count} ({safe_count / total_gaps * 100:.1f}%)\n"
        info_text += f"谨慎: {caution_count} ({caution_count / total_gaps * 100:.1f}%)\n"
        info_text += f"危险: {danger_count} ({danger_count / total_gaps * 100:.1f}%)"

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # 添加图例
    if caution_count > 0 or danger_count > 0:
        legend_elements = []
        if caution_count > 0:
            legend_elements.append(Line2D([0], [0], color='orange', linewidth=2, label='谨慎区域'))
        if danger_count > 0:
            legend_elements.append(Line2D([0], [0], color='red', linewidth=2, label='危险区域'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def run_visualization(config=None, show_danger_zones=False):
    """运行障碍物可视化

    Args:
        config: 配置字典
        show_danger_zones: 是否显示危险区域
    """
    stages = [
        TrainingStage.BASIC_FLIGHT,
        TrainingStage.SIMPLE_NAV,
        TrainingStage.COMPLEX_NAV,
        TrainingStage.GENERALIZATION
    ]
    stage_names = ['BASIC_FLIGHT', 'SIMPLE_NAV', 'COMPLEX_NAV', 'GENERALIZATION']

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, stage in enumerate(stages):
        # 根据阶段选择合适的区域大小
        area_sizes = {
            TrainingStage.BASIC_FLIGHT: (1000, 1000),
            TrainingStage.SIMPLE_NAV: (1000, 1000),
            TrainingStage.COMPLEX_NAV: (5000, 5000),
            TrainingStage.GENERALIZATION: (10000, 10000)  # 缩小以便可视化
        }

        area_size = area_sizes[stage]
        gen = ObstacleGenerator(area_size, stage, config=config)
        gen.generate(seed=42)

        print(f"Stage {stage_names[idx]}: Generated {len(gen.obstacles)} obstacles")

        # 选择可视化方式
        if show_danger_zones and stage in [TrainingStage.COMPLEX_NAV, TrainingStage.GENERALIZATION]:
            visualize_obstacles_with_danger(gen, stage_names[idx], axes[idx], config)
        else:
            visualize_obstacles(gen, stage_names[idx], axes[idx], config)

    title = 'Obstacle Distribution with Danger Zones' if show_danger_zones else 'Obstacle Distribution Across Training Stages'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    filename = 'obstacle_visualization_danger.png' if show_danger_zones else 'obstacle_visualization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Visualization saved as '{filename}'")

    plt.close(fig)
    return filename


def run_single_stage_visualization(stage: TrainingStage, config=None, show_danger=True):
    """单独可视化某个阶段（大图）"""
    area_sizes = {
        TrainingStage.BASIC_FLIGHT: (1000, 1000),
        TrainingStage.SIMPLE_NAV: (1000, 1000),
        TrainingStage.COMPLEX_NAV: (5000, 5000),
        TrainingStage.GENERALIZATION: (10000, 10000)
    }

    area_size = area_sizes[stage]
    gen = ObstacleGenerator(area_size, stage, config=config)
    gen.generate(seed=42)

    fig, ax = plt.subplots(figsize=(12, 12))

    if show_danger and stage in [TrainingStage.COMPLEX_NAV, TrainingStage.GENERALIZATION]:
        visualize_obstacles_with_danger(gen, f'{stage.name} - Detailed View', ax, config)
    else:
        visualize_obstacles(gen, f'{stage.name} - Detailed View', ax, config)

    plt.tight_layout()

    filename = f'{stage.name.lower()}_detailed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Detailed visualization saved as '{filename}'")

    plt.close(fig)
    return filename


def load_config():
    """加载配置文件"""
    try:
        import yaml
        config_path = 'config/default.yaml'

        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            print("Using default parameters")
            return None

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
        return config
    except ImportError:
        print("PyYAML not installed, using default parameters")
        return None
    except Exception as e:
        print(f"Could not load config: {e}")
        print("Using default parameters")
        return None


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print(" " * 15 + "障碍物生成与可视化系统")
    print(" " * 10 + "Obstacle Generation & Visualization")
    print("=" * 60)
    print()


def main():
    """主函数"""
    print_banner()

    # 加载配置
    print("Step 1: Loading configuration...")
    config = load_config()
    print()

    # 生成基础可视化
    print("Step 2: Generating basic visualization (4 stages)...")
    file1 = run_visualization(config, show_danger_zones=False)
    print()

    # 生成危险区域可视化
    print("Step 3: Generating danger zone visualization...")
    file2 = run_visualization(config, show_danger_zones=True)
    print()

    # 生成详细视图
    print("Step 4: Generating COMPLEX_NAV detailed view...")
    file3 = run_single_stage_visualization(TrainingStage.COMPLEX_NAV, config, show_danger=True)
    print()

    print("Step 5: Generating GENERALIZATION detailed view...")
    file4 = run_single_stage_visualization(TrainingStage.GENERALIZATION, config, show_danger=True)
    print()

    # 总结
    print("=" * 60)
    print("✓ All visualizations completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. {file1}")
    print(f"  2. {file2}")
    print(f"  3. {file3}")
    print(f"  4. {file4}")
    print("\nYou can find these PNG files in the current directory.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nProgram finished.")