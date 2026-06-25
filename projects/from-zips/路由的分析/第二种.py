import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import random
from torch.nn import functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据处理函数（增强错误处理）
def parse_file(file_path):
    timestamps = []
    monitors = {}
    direct_links = []
    indirect_links = []

    def split_and_convert(as_str):
        return [int(part) for part in as_str.replace("_", ",").split(",") if part.isdigit()]

    with open(file_path, 'r') as file:
        for line_idx, line in enumerate(tqdm(file, desc="解析文件")):
            parts = line.strip().split()
            if not parts:
                continue

            try:
                if parts[0] == 'T':
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        timestamps = [int(parts[1]), int(parts[2])]
                    else:
                        raise ValueError(f"无效的T条目格式，行号:{line_idx + 1}")
                elif parts[0] == 'M':
                    if len(parts) >= 4:
                        ip = parts[1]
                        as_num = parts[2]
                        key = parts[3]
                        as_num = int(as_num) if as_num.isdigit() else None
                        key = int(key) if key.isdigit() else None
                        if key is not None:
                            monitors[key] = (ip, as_num)
                    else:
                        raise ValueError(f"无效的M条目格式，行号:{line_idx + 1}")
                elif parts[0] == 'D':
                    if len(parts) >= 4:
                        from_as_list = split_and_convert(parts[1])
                        to_as_list = split_and_convert(parts[2])
                        monitor_keys = [int(k) for k in parts[3:] if k.isdigit()]
                        for fa in from_as_list:
                            for ta in to_as_list:
                                direct_links.append((fa, ta, monitor_keys))
                    else:
                        raise ValueError(f"无效的D条目格式，行号:{line_idx + 1}")
                elif parts[0] == 'I':
                    if len(parts) >= 5:
                        from_as_list = split_and_convert(parts[1])
                        to_as_list = split_and_convert(parts[2])
                        gap_length = int(parts[3]) if parts[3].isdigit() else 0
                        monitor_keys = [int(k) for k in parts[4:] if k.isdigit()]
                        for fa in from_as_list:
                            for ta in to_as_list:
                                indirect_links.append((fa, ta, gap_length, monitor_keys))
                    else:
                        raise ValueError(f"无效的I条目格式，行号:{line_idx + 1}")
            except Exception as e:
                print(f"解析错误，跳过该行: {line.strip()} \n错误信息: {str(e)}")
                continue

    return timestamps, monitors, direct_links, indirect_links

# 构建 AS 图
def build_as_graph(direct_links):
    print("构建 AS 图（仅包含直接连接的有向边）...")
    G = nx.DiGraph()
    for fa, ta, _ in direct_links:
        G.add_edge(fa, ta)
    print(f"AS 图构建完成，节点数: {len(G.nodes)}, 边数: {len(G.edges)}")
    return G

# 选取相关度高的节点（基于度中心性）
def select_subgraph(G, num_nodes=100, start_index=0):
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    end = start_index + num_nodes
    selected_nodes = [node for node, _ in sorted_nodes[start_index:end]]
    return G.subgraph(selected_nodes)

# 计算 AS 的路由统计信息
def calculate_route_stats(G):
    route_stats = {}
    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        route_count = in_degree + out_degree
        route_stats[node] = {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'route_count': route_count
        }
    return route_stats

# 刻画 AS 之间的水平邻居关系
def characterize_horizontal_relations(G):
    horizontal_relations = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        horizontal_relations[node] = neighbors
    return horizontal_relations

# 刻画 AS 之间的垂直商业关系（简单假设）
def characterize_vertical_relations(G, indirect_links):
    vertical_relations = {}
    subgraph_nodes = set(G.nodes())
    for fa, ta, gap_length, _ in indirect_links:
        if gap_length > 0 and fa in subgraph_nodes and ta in subgraph_nodes:
            if fa not in vertical_relations:
                vertical_relations[fa] = {}
            vertical_relations[fa][ta] = gap_length
    return vertical_relations

# 创新性分析：路由多样性分析
def analyze_route_diversity(G):
    route_diversity = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            diversity = len(set(neighbors)) / len(neighbors)
        else:
            diversity = 0
        route_diversity[node] = diversity
    return route_diversity

# 创新性分析：路由稳定性分析
def analyze_route_stability(G, indirect_links):
    route_stability = {}
    subgraph_nodes = set(G.nodes())
    for node in G.nodes():
        incoming_paths = []
        for fa, ta, gap_length, _ in indirect_links:
            if ta == node and fa in subgraph_nodes:
                incoming_paths.append((fa, gap_length))
        if incoming_paths:
            stability = len(set([path[0] for path in incoming_paths])) / len(incoming_paths)
        else:
            stability = 0
        route_stability[node] = stability
    return route_stability

# BEAM 模型（修复参数归一化问题）
class BEAM(nn.Module):
    def __init__(self, num_nodes, embedding_dim=128):
        super(BEAM, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.l = nn.Parameter(torch.randn(embedding_dim))
        self.r = nn.Parameter(torch.randn(embedding_dim))
        self.r.data = F.normalize(self.r.data, dim=0)

    def forward(self, u, v):
        xu = self.embeddings(u)
        xv = self.embeddings(v)
        p_score = torch.sum((xv - xu) * self.l * (xv - xu), dim=1)
        h_score = torch.sum((xv - xu) * self.r, dim=1)
        return -p_score + h_score

# 负采样函数
def sample_negative_edges(G, positive_edges, num_neg=10):
    nodes = list(G.nodes())
    neg_edges = []
    pos_set = set(positive_edges)
    while len(neg_edges) < len(positive_edges) * num_neg:
        u, v = random.choices(nodes, k=2)
        if u != v and (u, v) not in pos_set and (v, u) not in pos_set:
            neg_edges.append((u, v))
    return neg_edges

# 训练 BEAM 模型
def train_beam_model(G, embedding_dim=128, epochs=1000, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    model = BEAM(len(node_to_idx), embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    positive_edges = list(G.edges())
    print(f"开始训练 BEAM 模型，总轮数: {epochs}，学习率: {lr}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        neg_edges = sample_negative_edges(G, positive_edges)
        progress_bar = tqdm(positive_edges, desc=f"Epoch {epoch + 1}/{epochs}", unit="edge")
        for u, v in progress_bar:
            u_idx = torch.tensor([node_to_idx[u]], device=device)
            v_idx = torch.tensor([node_to_idx[v]], device=device)
            score_pos = model(u_idx, v_idx)

            neg_batch = random.sample(neg_edges, min(len(neg_edges), 10))
            u_neg = torch.tensor([node_to_idx[x[0]] for x in neg_batch], device=device)
            v_neg = torch.tensor([node_to_idx[x[1]] for x in neg_batch], device=device)
            score_neg = model(u_neg, v_neg)

            loss = -torch.mean(torch.log(torch.sigmoid(score_pos - score_neg)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": total_loss / (len(positive_edges) * (epoch + 1))})
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, 平均损失: {total_loss / len(positive_edges):.4f}")
    print("BEAM 模型训练完成")
    return model, node_to_idx, {i: n for i, n in enumerate(G.nodes())}

# 可视化嵌入向量（带 t-SNE 降维，修复 perplexity 错误）
def visualize_embeddings(model, node_to_idx, idx_to_node, save_folder):
    model.eval()
    embeddings = model.embeddings.weight.detach().cpu().numpy()
    n_samples = embeddings.shape[0]
    if n_samples <= 1:
        print("样本数不足，无法进行 t-SNE 降维")
        return

    perplexity = min(max(2, n_samples - 1), 50)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    try:
        embeddings_2d = tsne.fit_transform(embeddings)
    except ValueError as e:
        print(f"t-SNE 降维失败，尝试使用 PCA 替代...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    for idx, node in idx_to_node.items():
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f"AS{node}")
    plt.title('AS 嵌入向量可视化（基于 BEAM）')
    plt.xlabel('维度 1')
    plt.ylabel('维度 2')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='small')

    # 调整坐标轴范围
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)

    # 保存可视化图表并输出提示
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, 'embeddings_visualization.png')
    plt.savefig(save_path)
    print(f"已成功保存嵌入向量可视化图表至 {save_path}")
    plt.close()

# 可视化 AS 的路由行为和关系
def visualize_as_info(G, route_stats, horizontal_relations, vertical_relations, route_diversity, route_stability, save_folder):
    # 路由统计信息可视化
    plt.figure(figsize=(12, 6))
    nodes = list(G.nodes())
    in_degrees = [route_stats[node]['in_degree'] for node in nodes]
    out_degrees = [route_stats[node]['out_degree'] for node in nodes]
    route_counts = [route_stats[node]['route_count'] for node in nodes]

    plt.subplot(131)
    plt.bar(nodes, in_degrees)
    plt.title('入度')
    plt.xlabel('AS 编号')
    plt.ylabel('入度')
    plt.ylim(0, max(in_degrees) * 1.1 if in_degrees else 1)

    plt.subplot(132)
    plt.bar(nodes, out_degrees)
    plt.title('出度')
    plt.xlabel('AS 编号')
    plt.ylabel('出度')
    plt.ylim(0, max(out_degrees) * 1.1 if out_degrees else 1)

    plt.subplot(133)
    plt.bar(nodes, route_counts)
    plt.title('路由数量')
    plt.xlabel('AS 编号')
    plt.ylabel('路由数量')
    plt.ylim(0, max(route_counts) * 1.1 if route_counts else 1)

    plt.tight_layout()
    save_path = os.path.join(save_folder, 'route_stats_visualization.png')
    plt.savefig(save_path)
    print(f"已成功保存路由统计信息可视化图表至 {save_path}")
    plt.close()

    # 水平邻居关系可视化
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.title('水平邻居关系')
    save_path = os.path.join(save_folder, 'horizontal_relations_visualization.png')
    plt.savefig(save_path)
    print(f"已成功保存水平邻居关系可视化图表至 {save_path}")
    plt.close()

    # 垂直商业关系可视化
    plt.figure(figsize=(12, 8))
    vertical_G = nx.DiGraph()
    for fa, targets in vertical_relations.items():
        for ta, gap_length in targets.items():
            if fa in G.nodes() and ta in G.nodes():
                vertical_G.add_edge(fa, ta, weight=gap_length)
    pos = nx.spring_layout(vertical_G)
    nx.draw_networkx_nodes(vertical_G, pos)
    nx.draw_networkx_edges(vertical_G, pos, edge_color='r')
    labels = nx.get_edge_attributes(vertical_G, 'weight')
    nx.draw_networkx_edge_labels(vertical_G, pos, edge_labels=labels)
    nx.draw_networkx_labels(vertical_G, pos)
    plt.title('垂直商业关系')
    save_path = os.path.join(save_folder, 'vertical_relations_visualization.png')
    plt.savefig(save_path)
    print(f"已成功保存垂直商业关系可视化图表至 {save_path}")
    plt.close()

    # 路由多样性可视化
    plt.figure(figsize=(12, 6))
    nodes = list(G.nodes())
    diversities = [route_diversity[node] for node in nodes]
    plt.bar(nodes, diversities)
    plt.title('路由多样性')
    plt.xlabel('AS 编号')
    plt.ylabel('路由多样性')
    plt.ylim(0, max(diversities) * 1.1 if diversities else 1)
    save_path = os.path.join(save_folder, 'route_diversity_visualization.png')
    plt.savefig(save_path)
    print(f"已成功保存路由多样性可视化图表至 {save_path}")
    plt.close()

    # 路由稳定性可视化
    plt.figure(figsize=(12, 6))
    nodes = list(G.nodes())
    stabilities = [route_stability[node] for node in nodes]
    plt.bar(nodes, stabilities)
    plt.title('路由稳定性')
    plt.xlabel('AS 编号')
    plt.ylabel('路由稳定性')
    plt.ylim(0, max(stabilities) * 1.1 if stabilities else 1)
    save_path = os.path.join(save_folder, 'route_stability_visualization.png')
    plt.savefig(save_path)
    print(f"已成功保存路由稳定性可视化图表至 {save_path}")
    plt.close()

# 主函数
def main():
    file_path = 'cycle-aslinks.l7.t1.c008040.20200101.txt'
    save_folder = 'visualizations'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    timestamps, monitors, direct_links, indirect_links = parse_file(file_path)
    print(f"时间戳范围: {timestamps[0]} ~ {timestamps[1]}")
    print(f"监视器数量: {len(monitors)}, 直接连接数: {len(direct_links)}, 间接连接数: {len(indirect_links)}")

    G = build_as_graph(direct_links)

    # 选择前 15 个高相关节点
    subgraph = select_subgraph(G, num_nodes=15)
    print(f"子图节点数: {len(subgraph.nodes)}, 边数: {len(subgraph.edges)}")

    model, node_to_idx, idx_to_node = train_beam_model(subgraph, embedding_dim=32, epochs=500)
    visualize_embeddings(model, node_to_idx, idx_to_node, save_folder)

    route_stats = calculate_route_stats(subgraph)
    horizontal_relations = characterize_horizontal_relations(subgraph)
    vertical_relations = characterize_vertical_relations(subgraph, indirect_links)
    route_diversity = analyze_route_diversity(subgraph)
    route_stability = analyze_route_stability(subgraph, indirect_links)

    visualize_as_info(subgraph, route_stats, horizontal_relations, vertical_relations, route_diversity, route_stability, save_folder)

if __name__ == "__main__":
    main()