import hashlib
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# ===================== 全局绘图设置（解决中文乱码） =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 密码学工具函数 =====================
def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def generate_keypair(node_id: int):
    secret_key = f"sk_node_{node_id}".encode()
    public_key = sha256(secret_key)
    return secret_key, public_key


def sign(secret_key: bytes, data: bytes) -> bytes:
    return sha256(secret_key + data)


def verify_sig(public_key: bytes, data: bytes, signature: bytes) -> bool:
    return signature == sha256(b"sk_node_" + str(public_key_to_id(public_key)).encode() + data)


def public_key_to_id(pk: bytes) -> int:
    for i in range(100):
        if sha256(f"sk_node_{i}".encode()) == pk:
            return i
    return -1


# ===================== 节点类 =====================
class Node:
    def __init__(self, node_id: int, is_honest: bool = True, value: bytes = None, k_bits: int = 256):
        self.node_id = node_id
        self.is_honest = is_honest
        self.sk, self.pk = generate_keypair(node_id)

        if value is None:
            value = random.randbytes(k_bits // 8)
        self.original_value = value
        self.commitment = sha256(value)

        self.commit_pool = {}
        self.prepare_msgs = []
        self.commit_msgs = []
        self.final_commitment = None
        self.final_value = None
        self.dac = None

        self.bytes_sent = 0
        self.bytes_received = 0

    # 阶段1：广播承诺
    def broadcast_commitment(self, all_nodes):
        msg = {
            'type': 'COMMITMENT',
            'node_id': self.node_id,
            'commitment': self.commitment,
            'signature': sign(self.sk, self.commitment)
        }
        msg_size = len(msg['commitment']) + len(msg['signature']) + 4
        self.bytes_sent += msg_size * len(all_nodes)

        for node in all_nodes:
            if node.node_id != self.node_id:
                node.receive_commitment(msg)
                node.bytes_received += msg_size

    def receive_commitment(self, msg):
        if not verify_sig(sha256(f"sk_node_{msg['node_id']}".encode()), msg['commitment'], msg['signature']):
            return
        self.commit_pool[msg['node_id']] = msg['commitment']

    # 阶段2：主节点提议
    def propose(self, view: int, all_nodes):
        if self.is_honest:
            c_star = self.commitment
        else:
            c_star = sha256(f"fake_{self.node_id}".encode())

        propose_msg = {
            'type': 'PROPOSE',
            'view': view,
            'commitment': c_star,
            'primary_id': self.node_id,
            'signature': sign(self.sk, c_star)
        }
        msg_size = len(c_star) + len(propose_msg['signature']) + 8
        self.bytes_sent += msg_size * len(all_nodes)

        for node in all_nodes:
            node.receive_propose(propose_msg, all_nodes)
            node.bytes_received += msg_size

    def receive_propose(self, msg, all_nodes):
        primary_pk = sha256(f"sk_node_{msg['primary_id']}".encode())
        if not verify_sig(primary_pk, msg['commitment'], msg['signature']):
            return

        c_star = msg['commitment']
        has_value = (self.commitment == c_star)

        if not has_value:
            primary = next(n for n in all_nodes if n.node_id == msg['primary_id'])
            value = primary.provide_value(c_star)
            if value and sha256(value) == c_star:
                self.original_value = value
                has_value = True
                self.bytes_received += len(value)
                primary.bytes_sent += len(value)

        if has_value and self.is_honest:
            prepare_msg = {
                'type': 'PREPARE',
                'node_id': self.node_id,
                'commitment': c_star,
                'signature': sign(self.sk, c_star)
            }
            msg_size = len(c_star) + len(prepare_msg['signature']) + 4
            self.bytes_sent += msg_size * len(all_nodes)

            for node in all_nodes:
                if node.node_id != self.node_id:
                    node.receive_prepare(prepare_msg)
                    node.bytes_received += msg_size

    def provide_value(self, commitment: bytes) -> bytes:
        if self.is_honest and self.commitment == commitment:
            return self.original_value
        elif not self.is_honest:
            if random.random() < 0.5:
                return None
            else:
                return random.randbytes(len(self.original_value))
        return None

    # 阶段3：准备阶段
    def receive_prepare(self, msg):
        sender_pk = sha256(f"sk_node_{msg['node_id']}".encode())
        if not verify_sig(sender_pk, msg['commitment'], msg['signature']):
            return
        if not any(m['node_id'] == msg['node_id'] for m in self.prepare_msgs):
            self.prepare_msgs.append(msg)

    def check_prepared(self, f: int) -> bool:
        return len(self.prepare_msgs) >= 2 * f

    # 阶段4：提交阶段
    def broadcast_commit_message(self, f: int, all_nodes):
        if not self.check_prepared(f):
            return

        c_star = self.prepare_msgs[0]['commitment']
        dac_candidate = [m['signature'] for m in self.prepare_msgs[:f + 1]]

        commit_msg = {
            'type': 'COMMIT',
            'node_id': self.node_id,
            'commitment': c_star,
            'dac': dac_candidate,
            'signature': sign(self.sk, c_star)
        }
        msg_size = len(c_star) + sum(len(s) for s in dac_candidate) + len(commit_msg['signature']) + 4
        self.bytes_sent += msg_size * len(all_nodes)

        for node in all_nodes:
            if node.node_id != self.node_id:
                node.receive_commit_message(commit_msg)
                node.bytes_received += msg_size

    def receive_commit_message(self, msg):
        sender_pk = sha256(f"sk_node_{msg['node_id']}".encode())
        if not verify_sig(sender_pk, msg['commitment'], msg['signature']):
            return
        if not any(m['node_id'] == msg['node_id'] for m in self.commit_msgs):
            self.commit_msgs.append(msg)

    def check_committed(self, f: int) -> bool:
        return len(self.commit_msgs) >= 2 * f

    # 阶段5：数据恢复
    def recover_value(self, f: int, all_nodes):
        if not self.check_committed(f):
            return False

        self.final_commitment = self.commit_msgs[0]['commitment']

        if self.commitment == self.final_commitment:
            self.final_value = self.original_value
            return True

        commiters = [m['node_id'] for m in self.commit_msgs[:f + 1]]
        for node_id in commiters:
            node = next(n for n in all_nodes if n.node_id == node_id)
            value = node.provide_value(self.final_commitment)
            if value and sha256(value) == self.final_commitment:
                self.final_value = value
                self.bytes_received += len(value)
                node.bytes_sent += len(value)
                return True

        return False


# ===================== 共识模拟器 =====================
class ConsensusSimulator:
    def __init__(self, N: int, f: int, k_bits: int = 1024):
        assert N >= 3 * f + 1, "必须满足 N ≥ 3f + 1"
        self.N = N
        self.f = f
        self.k_bits = k_bits
        self.nodes = []
        self.view = 0

    def setup_nodes(self, byzantine_ids=None):
        if byzantine_ids is None:
            byzantine_ids = random.sample(range(self.N), self.f)

        self.nodes = []
        for i in range(self.N):
            is_honest = i not in byzantine_ids
            node = Node(i, is_honest=is_honest, k_bits=self.k_bits)
            self.nodes.append(node)

        return byzantine_ids

    def run_round(self) -> dict:
        # 阶段1：广播承诺
        for node in self.nodes:
            node.broadcast_commitment(self.nodes)

        # 选主节点
        primary_id = self.view % self.N
        primary = self.nodes[primary_id]

        # 阶段2：提议
        primary.propose(self.view, self.nodes)
        time.sleep(0.001)

        # 阶段3-4：准备与提交
        for node in self.nodes:
            if node.is_honest and node.check_prepared(self.f):
                node.broadcast_commit_message(self.f, self.nodes)
        time.sleep(0.001)

        # 阶段5：数据恢复
        success_count = 0
        honest_nodes = [n for n in self.nodes if n.is_honest]
        for node in honest_nodes:
            success = node.recover_value(self.f, self.nodes)
            if success:
                success_count += 1

        metrics = self._calculate_metrics(honest_nodes)
        metrics['recovery_success_rate'] = success_count / len(honest_nodes)
        metrics['byzantine_count'] = self.f
        metrics['node_count'] = self.N
        return metrics

    def _calculate_metrics(self, honest_nodes) -> dict:
        values = [n.final_value for n in honest_nodes if n.final_value is not None]
        consistency = len(set(values)) == 1 and len(values) == len(honest_nodes)

        total_bytes_hcdac = sum(n.bytes_sent for n in self.nodes)
        traditional_bytes = self.N * self.N * (self.k_bits // 8) * 3
        compression_ratio = traditional_bytes / total_bytes_hcdac if total_bytes_hcdac > 0 else 0

        verifiable = all(
            sha256(n.final_value) == n.final_commitment
            for n in honest_nodes if n.final_value is not None
        )

        return {
            'consistency': consistency,
            'compression_ratio': round(compression_ratio, 2),
            'verifiable': verifiable,
            'total_bytes_hcdac': total_bytes_hcdac,
            'traditional_bytes': traditional_bytes
        }


# ===================== 绘图函数 =====================
def plot_all_charts():
    # 基础参数
    N_base = 7
    f_base = 2
    k_base = 8192
    rounds = 10

    # ========== 图1：不同k值下的带宽压缩比 ==========
    k_list = [256, 1024, 4096, 16384, 65536, 262144]
    compression_ratios = []
    save_ratios = []

    for k in k_list:
        sim = ConsensusSimulator(N_base, f_base, k)
        sim.setup_nodes()
        m = sim.run_round()
        compression_ratios.append(m['compression_ratio'])
        save_ratios.append((1 - 1 / m['compression_ratio']) * 100 if m['compression_ratio'] > 0 else 0)

    # ========== 图2：传统方案 vs HC-DAC 通信量对比 ==========
    k_labels = ['1KB', '4KB', '16KB', '64KB']
    k_values = [8192, 32768, 131072, 524288]
    traditional = []
    hcdac = []

    for k in k_values:
        sim = ConsensusSimulator(N_base, f_base, k)
        sim.setup_nodes()
        m = sim.run_round()
        traditional.append(m['traditional_bytes'] / 1024)  # 转KB
        hcdac.append(m['total_bytes_hcdac'] / 1024)

    # ========== 图3：不同拜占庭节点数下的共识成功率 ==========
    N_test = 10
    f_list = list(range(0, 4))  # f=0,1,2,3 （N=10最大f=3）
    consistency_rates = []
    recovery_rates = []

    for f in f_list:
        consis_sum = 0
        recover_sum = 0
        for _ in range(rounds):
            sim = ConsensusSimulator(N_test, f, k_base)
            sim.setup_nodes()
            m = sim.run_round()
            consis_sum += m['consistency']
            recover_sum += m['recovery_success_rate']
        consistency_rates.append(consis_sum / rounds * 100)
        recovery_rates.append(recover_sum / rounds * 100)

    # ========== 图4：不同节点总数下的通信开销 ==========
    N_list = [4, 7, 10, 13, 16]
    f_map = {4: 1, 7: 2, 10: 3, 13: 4, 16: 5}
    total_bytes_list = []

    for N in N_list:
        f = f_map[N]
        sim = ConsensusSimulator(N, f, k_base)
        sim.setup_nodes()
        m = sim.run_round()
        total_bytes_list.append(m['total_bytes_hcdac'] / 1024)

    # ========== 绘制2x2子图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('基于哈希承诺与DAC的多值拜占庭共识算法 性能可视化', fontsize=14, fontweight='bold')

    # 子图1：带宽压缩比
    ax1 = axes[0][0]
    x = range(len(k_list))
    ax1.bar(x, compression_ratios, color='#1f77b4', alpha=0.7, label='带宽压缩比')
    ax1.plot(x, save_ratios, color='#ff7f0e', marker='o', linewidth=2, label='带宽节省率(%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{k // 8}B' if k < 8192 else f'{k // 8 // 1024}KB' for k in k_list])
    ax1.set_title('不同原始值长度下的带宽压缩效果')
    ax1.set_xlabel('原始值大小')
    ax1.set_ylabel('压缩倍数 / 节省比例(%)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 子图2：通信量对比
    ax2 = axes[0][1]
    x = range(len(k_labels))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], traditional, width, label='传统全量传输方案', color='#d62728', alpha=0.7)
    ax2.bar([i + width / 2 for i in x], hcdac, width, label='HC-DAC方案', color='#2ca02c', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(k_labels)
    ax2.set_title('两种方案单轮总通信量对比')
    ax2.set_xlabel('原始值大小')
    ax2.set_ylabel('总通信量 (KB)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 子图3：拜占庭容错能力
    ax3 = axes[1][0]
    ax3.plot(f_list, consistency_rates, marker='s', linewidth=2, label='一致性达成率', color='#9467bd')
    ax3.plot(f_list, recovery_rates, marker='^', linewidth=2, label='数据恢复成功率', color='#8c564b')
    ax3.set_title(f'不同拜占庭节点数下的共识表现 (N={N_test})')
    ax3.set_xlabel('拜占庭节点数量 f')
    ax3.set_ylabel('成功率 (%)')
    ax3.set_ylim(80, 105)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 子图4：节点扩展性
    ax4 = axes[1][1]
    ax4.plot(N_list, total_bytes_list, marker='o', linewidth=2, color='#e377c2')
    ax4.set_title('不同节点规模下的总通信开销')
    ax4.set_xlabel('总节点数 N (满足N≥3f+1)')
    ax4.set_ylabel('总通信量 (KB)')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('共识算法性能图表.png', dpi=300, bbox_inches='tight')
    print("✅ 图表已保存为：共识算法性能图表.png")
    plt.show()


# ===================== 主程序 =====================
if __name__ == "__main__":
    N = 7
    f = 2
    k_bits = 8192

    print(f"=== 测试配置：N={N}, f={f}, 原始值长度={k_bits}比特 ===")
    print(f"满足 N ≥ 3f+1：{N >= 3 * f + 1}\n")

    rounds = 20
    all_metrics = defaultdict(list)

    for r in range(rounds):
        sim = ConsensusSimulator(N, f, k_bits)
        sim.setup_nodes()
        metrics = sim.run_round()
        for k, v in metrics.items():
            all_metrics[k].append(v)

    print("=" * 50)
    print("算法验证指标统计（20轮平均）")
    print("=" * 50)
    print(f"1. 一致性达成率：{sum(all_metrics['consistency']) / rounds * 100:.1f}%")
    print(f"2. 数据恢复成功率：{sum(all_metrics['recovery_success_rate']) / rounds * 100:.1f}%")
    print(f"3. 哈希可验证率：{sum(all_metrics['verifiable']) / rounds * 100:.1f}%")
    print(f"4. 通信带宽压缩比：{sum(all_metrics['compression_ratio']) / rounds:.2f} 倍")
    print(f"   - 传统方案平均通信量：{sum(all_metrics['traditional_bytes']) / rounds:.0f} 字节")
    print(f"   - HC-DAC方案平均通信量：{sum(all_metrics['total_bytes_hcdac']) / rounds:.0f} 字节")
    print("=" * 50)

    # 生成并展示图表
    print("\n正在生成可视化图表...")
    plot_all_charts()