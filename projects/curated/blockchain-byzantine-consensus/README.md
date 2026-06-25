# 区块链拜占庭共识仿真

## 项目简介

基于 Python 的多值拜占庭共识协议（MVBC）仿真。核心协议 Hash-DAC 通过数据可用性委员会（DAC）+ Merkle Tree 证明 + Reed-Solomon 纠删码，实现在 f 个拜占庭节点下的安全共识，避免反复传输完整原始数据。

## 代码架构

### Hash-DAC 共识协议 (`mvbc_hash_dac.py`)

```python
# 安全假设: N >= 3f + 1, 最多 f 个拜占庭节点
# CandidateID = (proposer, value_hash, merkle_root, data_len, N, f, threshold)

# 协议阶段:
#   PROPOSE: 提议者 → CandidateID + Reed-Solomon 纠删码分片
#   PREPARE:  节点验证 Hash 承诺 + Merkle Proof → 投票
#   COMMIT:   收集 2f+1 Prepare 投票 → 锁定 CandidateID
#   DECIDE:   纠删码恢复 (任意 k 个有效分片) → Hash 验证一致性

# 有限域运算 (GF(257)):
def mod_inv(a):       # 模逆: pow(a, PRIME-2, 257)
def gf_eval_poly(c,x):# 多项式求值 → Reed-Solomon 编码
def lagrange_interpolate(points): # 拉格朗日插值 → 纠删码恢复
```

### Merkle Tree 数据可用性证明

```
原始数据 → 分块 [b0, b1, ..., bk-1]
  → sha256 叶节点 → 逐层配对 hash → Merkle Root
Merkle Proof: 叶节点 + 兄弟路径 hash 链
验证: proof + leaf_hash → recompute root → == merkle_root ?
```

### 纠删码 (Reed-Solomon over GF(257))

```python
# 编码: k 块数据 → n 块编码 (容忍 n-k 丢失)
#   coeffs = [data[i] for i in range(k)]
#   shards[i] = gf_eval_poly(coeffs, i+1)  # i ∈ [0, n)
#
# 恢复: 任意 k 个有效分片 → 拉格朗日插值多项式 → 原始数据
```

### 拜占庭行为模拟

- **错误分片**: 发送被篡改/随机分片数据
- **矛盾投票**: 对同一轮发送不同的 Prepare/Commit
- **静默**: 故意不投票，测试 liveness
- **伪造证明**: 发送错误的 Merkle Proof

### PBFT 对比仿真 (`basic_consensus_simulation.py`)

```python
# 模拟签名: sha256(secret || message) → 真实系统应替换为 Ed25519
# 三阶段: Pre-Prepare → Prepare → Commit
# 性能对比指标: 吞吐量, 通信消息数, 延迟 (共识达成时间)
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 共识协议 | Hash-DAC, PBFT, MVBC |
| 密码学 | SHA-256, 模拟签名, Merkle Tree + Proof |
| 纠删码 | Reed-Solomon GF(257) |
| 拜占庭模型 | f < N/3, 任意行为 (错误/矛盾/静默/伪造) |

## 运行方式

```bash
pip install numpy matplotlib
python mvbc_hash_dac.py              # Hash-DAC 协议完整仿真
python basic_consensus_simulation.py # PBFT 对比实验
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `mvbc_hash_dac.py` | Hash-DAC 协议: CandidateID + RS编码 + Merkle + 共识 |
| `basic_consensus_simulation.py` | PBFT 风格三阶段共识对比 |
| `consensus_performance.png` | 共识协议性能对比图 |
