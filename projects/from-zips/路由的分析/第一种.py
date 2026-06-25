import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import os
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from community import best_partition  # 需安装 python-louvain


# 创建保存图片的文件夹
def create_output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"as_visualizations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# === 步骤 1: 加载并解析 CAIDA AS-links 文件 ===
def load_as_links(filename):
    monitors = {}
    direct_links = []
    print("[步骤 1] 读取并解析 AS 链接文件...")

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('M'):
                _, ip, asn, key = line.strip().split()
                monitors[int(key)] = int(asn)
            elif line.startswith('D'):
                parts = line.strip().split()
                from_as_list = parts[1].split(',')
                to_as_list = parts[2].split(',')
                for from_as_str in from_as_list:
                    for to_as_str in to_as_list:
                        try:
                            from_as = int(from_as_str)
                            to_as = int(to_as_str)
                            direct_links.append((from_as, to_as))
                        except ValueError:
                            print(f"跳过非法字段: from_as={from_as_str}, to_as={to_as_str}")

    print(f" 监视器数: {len(monitors)}，有效链接数: {len(direct_links)}")
    return monitors, direct_links


# === 步骤 2: 构建有向 AS 图 ===
def build_as_graph(direct_links):
    print("[步骤 2] 构建有向 AS 图...")
    G = nx.DiGraph()
    for u, v in direct_links:
        G.add_edge(u, v)
    print(f" 图中节点数: {G.number_of_nodes()}，边数: {G.number_of_edges()}")
    return G


# === 步骤 3: 按度数选择顶级节点 ===
def select_top_nodes_by_degree(G, start_from=5, top_k=10):
    print(f"[步骤 3] 使用度中心性选择从第 {start_from} 个开始的 {top_k} 个节点...")
    degree_scores = G.degree()
    sorted_nodes = sorted(degree_scores, key=lambda x: x[1], reverse=True)
    selected = [node for node, _ in sorted_nodes[start_from:start_from + top_k]]
    print(f" 选择节点: {selected}")
    return selected


# === 步骤 4: 过滤图，仅保留选定节点 ===
def filter_graph_by_nodes(G, selected_nodes):
    print("[步骤 4] 构建选定节点的子图...")
    subgraph = G.subgraph(selected_nodes).copy()
    print(f" 子图节点数: {subgraph.number_of_nodes()}，边数: {subgraph.number_of_edges()}")
    return subgraph


# === 步骤 5: 计算邻近性指标 ===
def compute_proximities(G, use_cuda=True):
    print("[步骤 5] 计算邻近性指标（一阶 & 二阶）...")

    nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    print(f" 节点总数: {n}")

    adj_matrix = torch.zeros((n, n), dtype=torch.float32)
    for u, v in G.edges():
        i, j = node_idx[u], node_idx[v]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(" 使用设备:", device)
    adj_matrix = adj_matrix.to(device)

    first_order = {}
    second_order = {}

    for i in range(n):
        vi = adj_matrix[i]
        for j in range(i + 1, n):
            vj = adj_matrix[j]
            first_order_score = adj_matrix[i, j].item()
            intersection = torch.sum(torch.min(vi, vj)).item()
            union = torch.sum(torch.max(vi, vj)).item()
            second_order_score = (intersection / union) if union > 0 else 0.0
            first_order[(nodes[i], nodes[j])] = first_order_score
            second_order[(nodes[i], nodes[j])] = second_order_score

    print(" 一阶 & 二阶邻近性计算完成 ✅")
    return first_order, second_order


# === 步骤 6: 定义 BEAM 模型 ===
class BEAM(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(BEAM, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, src, dst):
        src_emb = self.embeddings(src)
        dst_emb = self.embeddings(dst)
        return torch.norm(src_emb - dst_emb, p=2, dim=1)


# === 步骤 7: 训练 BEAM 模型 ===
def train_beam(G, first_order, second_order, epochs=100, embedding_dim=64, lr=0.01):
    print("[步骤 6] 训练 BEAM 模型...")
    node2idx = {node: idx for idx, node in enumerate(G.nodes())}
    model = BEAM(len(node2idx), embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    pairs = list(first_order.keys())
    targets = torch.tensor([first_order[p] + second_order[p] for p in pairs], dtype=torch.float)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        src = torch.tensor([node2idx[p[0]] for p in pairs])
        dst = torch.tensor([node2idx[p[1]] for p in pairs])
        distances = model(src, dst)
        loss = loss_fn(distances, targets)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

    print(" BEAM 模型训练完成 ✅")
    return model, node2idx


# === 步骤 8: 提取嵌入向量 ===
def extract_embeddings(model, node2idx):
    print("[步骤 7] 提取嵌入向量...")
    with torch.no_grad():
        embeddings = model.embeddings.weight.data.cpu().numpy()
        print(f" 嵌入维度: {embeddings.shape[1]}，节点数: {len(node2idx)}")
        return {node: embeddings[idx] for node, idx in node2idx.items()}


# === 步骤 9: 分析 AS 关系 ===
def analyze_as_relationships(G, embeddings, first_order, second_order):
    print("[步骤 8] 分析 AS 路由行为和关系...")

    nodes = list(G.nodes())
    degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G)

    relationships = {
        '节点': [],
        '度数': [],
        '入度': [],
        '出度': [],
        '介数中心性': [],
        '类型': [],
        '路由活跃度': [],
        '水平邻居得分': [],
        '垂直商业得分': []
    }

    for node in nodes:
        degree = degrees.get(node, 0)
        in_degree = in_degrees.get(node, 0)
        out_degree = out_degrees.get(node, 0)
        betweenness_score = betweenness.get(node, 0)
        routing_activity = degree / max(degrees.values()) if degrees else 0

        horizontal_sum = sum(second_order.get((node, n), 0) for n in nodes if n != node)
        horizontal_score = horizontal_sum / (len(nodes) - 1) if len(nodes) > 1 else 0

        vertical_sum = sum(first_order.get((node, n), 0) *
                           abs(degrees.get(node, 0) - degrees.get(n, 0))
                           for n in nodes if n != node)
        vertical_score = vertical_sum / (len(nodes) - 1) if len(nodes) > 1 else 0

        if degree > 3:
            rel_type = "商业上游关系"
        elif degree == 3:
            rel_type = "邻居关系"
        else:
            rel_type = "路由观察点"

        relationships['节点'].append(node)
        relationships['度数'].append(degree)
        relationships['入度'].append(in_degree)
        relationships['出度'].append(out_degree)
        relationships['介数中心性'].append(betweenness_score)
        relationships['类型'].append(rel_type)
        relationships['路由活跃度'].append(routing_activity)
        relationships['水平邻居得分'].append(horizontal_score)
        relationships['垂直商业得分'].append(vertical_score)

    return pd.DataFrame(relationships)


# === 步骤 10: 可视化 AS 分析 ===
def visualize_as_analysis(embeddings, G, first_order, second_order, output_dir):
    print("[步骤 9] 可视化 AS 路由行为和关系...")

    # 设置 Seaborn 风格
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # 确保中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 提取嵌入并降维
    nodes = list(embeddings.keys())
    X = np.array([embeddings[n] for n in nodes])
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_2d = tsne.fit_transform(X)

    # 分析关系
    df = analyze_as_relationships(G, embeddings, first_order, second_order)
    df['TSNE-1'] = X_2d[:, 0]
    df['TSNE-2'] = X_2d[:, 1]

    # 调试：打印数据框列名
    print("数据框列名:", df.columns.tolist())

    # 保存统计到 CSV
    df[['节点', '入度', '出度', '度数', '介数中心性', '类型', '路由活跃度', '水平邻居得分', '垂直商业得分']].to_csv(
        os.path.join(output_dir, 'degree_stats.csv'), index=False, encoding='utf-8-sig'
    )
    print("统计数据已保存至: degree_stats.csv")

    # 1. AS 路由行为分布（按类型）
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', hue='类型',
                    size='路由活跃度', sizes=(50, 500),
                    style='类型', palette='Set2', alpha=0.7, ax=ax)
    ax.set_title('AS 路由行为分布（按类型）', fontsize=16)
    ax.set_xlabel('TSNE 维度 1', fontsize=12)
    ax.set_ylabel('TSNE 维度 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='类型')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'as_routing_behavior.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. 水平邻居关系分布
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2',
                    hue='水平邻居得分', size='度数',
                    sizes=(50, 500), palette='viridis', alpha=0.7, ax=ax)
    ax.set_title('水平邻居关系分布', fontsize=16)
    ax.set_xlabel('TSNE 维度 1', fontsize=12)
    ax.set_ylabel('TSNE 维度 2', fontsize=12)
    norm = Normalize(vmin=df['水平邻居得分'].min(), vmax=df['水平邻居得分'].max())
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    fig.colorbar(sm, ax=ax, label='水平邻居得分')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'horizontal_relationships.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. 垂直商业关系分布
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2',
                    hue='垂直商业得分', size='度数',
                    sizes=(50, 500), palette='magma', alpha=0.7, ax=ax)
    ax.set_title('垂直商业关系分布', fontsize=16)
    ax.set_xlabel('TSNE 维度 1', fontsize=12)
    ax.set_ylabel('TSNE 维度 2', fontsize=12)
    norm = Normalize(vmin=df['垂直商业得分'].min(), vmax=df['垂直商业得分'].max())
    sm = cm.ScalarMappable(cmap='magma', norm=norm)
    fig.colorbar(sm, ax=ax, label='垂直商业得分')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vertical_relationships.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4. 指标相关性热图
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[['路由活跃度', '水平邻居得分', '垂直商业得分', '度数', '入度', '出度', '介数中心性']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('AS 指标相关性热图', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 5. 水平邻居关系有向图
    fig, ax = plt.subplots(figsize=(12, 8))
    H = nx.DiGraph()
    H.add_nodes_from(nodes)
    degrees = dict(G.degree())
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:
                score = second_order.get((node1, node2), 0)
                if score > 0.1:  # 阈值过滤弱关系
                    H.add_edge(node1, node2, weight=score)

    pos = nx.spring_layout(H)
    node_sizes = [degrees.get(node, 1) * 100 for node in H.nodes()]
    edge_weights = [H[u][v]['weight'] * 5 for u, v in H.edges()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color='lightblue', ax=ax)
    nx.draw_networkx_edges(H, pos, width=edge_weights, arrows=True, ax=ax)
    nx.draw_networkx_labels(H, pos, font_family='SimHei', font_size=10, ax=ax)
    ax.set_title('水平邻居关系有向图', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'horizontal_network.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 6. 垂直商业关系有向图
    fig, ax = plt.subplots(figsize=(12, 8))
    V = nx.DiGraph()
    V.add_nodes_from(nodes)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:
                score = first_order.get((node1, node2), 0)  # 仅使用 first_order 得分
                if score > 0.1:  # 阈值过滤弱关系
                    V.add_edge(node1, node2, weight=score)

    pos = nx.spring_layout(V)
    node_sizes = [degrees.get(node, 1) * 100 for node in V.nodes()]
    edge_weights = [V[u][v]['weight'] * 5 for u, v in V.edges()]
    nx.draw_networkx_nodes(V, pos, node_size=node_sizes, node_color='lightgreen', ax=ax)
    nx.draw_networkx_edges(V, pos, width=edge_weights, arrows=True, ax=ax)
    nx.draw_networkx_labels(V, pos, font_family='SimHei', font_size=10, ax=ax)
    ax.set_title('垂直商业关系有向图', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vertical_network.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 7. 入度 vs 出度分布
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='入度', y='出度', hue='类型',
                    size='度数', sizes=(50, 500), palette='Set2', alpha=0.7, ax=ax)
    ax.set_title('入度 vs 出度分布', fontsize=16)
    ax.set_xlabel('入度', fontsize=12)
    ax.set_ylabel('出度', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='类型')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 8. 中心性分析
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='介数中心性', y='度数', hue='类型',
                    size='路由活跃度', sizes=(50, 500), palette='Set2', alpha=0.7, ax=ax)
    ax.set_title('介数中心性 vs 度数分布', fontsize=16)
    ax.set_xlabel('介数中心性', fontsize=12)
    ax.set_ylabel('度数', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='类型')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'centrality_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 9. 社区检测有向图
    fig, ax = plt.subplots(figsize=(12, 8))
    G_undirected = G.to_undirected()  # Louvain 需要无向图
    communities = best_partition(G_undirected)
    community_colors = [communities[node] for node in nodes]
    pos = nx.spring_layout(G)
    node_sizes = [degrees.get(node, 1) * 100 for node in nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=community_colors, cmap='tab10', ax=ax)
    nx.draw_networkx_edges(G, pos, width=1, arrows=True, ax=ax)
    nx.draw_networkx_labels(G, pos, font_family='SimHei', font_size=10, ax=ax)
    ax.set_title('社区检测有向图', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'community_network.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 社区规模统计
    community_sizes = pd.Series(communities).value_counts()
    print("社区规模统计:", community_sizes.to_dict())

    # 10. 嵌入向量分布
    fig, ax = plt.subplots(figsize=(12, 8))
    embedding_values = np.array([embeddings[node] for node in nodes])
    ax.boxplot(embedding_values, vert=True, patch_artist=True)
    ax.set_title('嵌入向量维度分布', fontsize=16)
    ax.set_xlabel('维度', fontsize=12)
    ax.set_ylabel('值', fontsize=12)
    ax.set_xticks(range(1, embedding_values.shape[1] + 1, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"所有可视化图表已保存至: {output_dir}")


# === 主程序 ===
if __name__ == "__main__":
    file_path = "cycle-aslinks.l7.t1.c000027.20070913.txt"

    # 创建输出文件夹
    output_dir = create_output_folder()

    # 执行步骤
    monitors, links = load_as_links(file_path)
    full_graph = build_as_graph(links)
    selected_nodes = select_top_nodes_by_degree(full_graph, start_from=10, top_k=15)
    subgraph = filter_graph_by_nodes(full_graph, selected_nodes)
    f_prox, s_prox = compute_proximities(subgraph)
    model, node2idx = train_beam(subgraph, f_prox, s_prox)
    embeddings = extract_embeddings(model, node2idx)

    # 执行可视化
    visualize_as_analysis(embeddings, subgraph, f_prox, s_prox, output_dir)

    print("[完成] 所有分析和可视化已完成！")