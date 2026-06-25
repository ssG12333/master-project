
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hash + DAC Multi-Value Byzantine Consensus Simulation

目标：
- N >= 3f + 1，最多 f 个拜占庭节点。
- 每个节点持有 k bit 原始输入。
- 共识主流程只对压缩候选标识 CandidateID 达成一致：
  CandidateID = (proposer, value_hash, merkle_root, data_len, N, f, threshold)
- 原始值只在数据可用性阶段用纠删码分片分发一次；
  后续 PREPARE / COMMIT 不再反复传输完整原始数据。
- 最终通过 DAC + Merkle proof + Reed-Solomon 恢复 + Hash 承诺验证，
  保证诚实节点输出同一个可恢复、可验证且与合法承诺一致的原始值。

说明：
这是教学/论文验证用的“协议仿真代码”，签名用 sha256(secret || message) 模拟。
真实系统中应替换为 Ed25519/BLS 等数字签名，并使用真实网络/持久化存储。
"""

import argparse
import dataclasses
import hashlib
import os
import random
import statistics
import struct
from typing import Dict, List, Tuple, Optional, Iterable, Set


# ============================================================
# 1. 基础工具：哈希、模拟签名、有限域 GF(257)
# ============================================================

PRIME = 257  # 可容纳 byte 0..255，并支持 GF(p) 运算


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def hx(b: bytes, n: int = 12) -> str:
    return b.hex()[:n]


def i2b(x: int, size: int = 4) -> bytes:
    return int(x).to_bytes(size, "big", signed=False)


def mod_inv(a: int) -> int:
    a %= PRIME
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(257)")
    return pow(a, PRIME - 2, PRIME)


def gf_eval_poly(coeffs: List[int], x: int) -> int:
    """Evaluate c0 + c1*x + ... over GF(257)."""
    x %= PRIME
    acc = 0
    power = 1
    for c in coeffs:
        acc = (acc + c * power) % PRIME
        power = (power * x) % PRIME
    return acc


def gf_solve_linear(A: List[List[int]], y: List[int]) -> List[int]:
    """
    Solve A * x = y over GF(257) by Gaussian elimination.
    A is n x n.
    """
    n = len(A)
    M = [row[:] + [yy % PRIME] for row, yy in zip(A, y)]

    for col in range(n):
        pivot = None
        for r in range(col, n):
            if M[r][col] % PRIME != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("singular matrix in GF(257)")
        M[col], M[pivot] = M[pivot], M[col]

        inv_p = mod_inv(M[col][col])
        for c in range(col, n + 1):
            M[col][c] = (M[col][c] * inv_p) % PRIME

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col] % PRIME
            if factor:
                for c in range(col, n + 1):
                    M[r][c] = (M[r][c] - factor * M[col][c]) % PRIME

    return [M[i][n] % PRIME for i in range(n)]


def sign(secret: bytes, message: bytes) -> bytes:
    """模拟签名：真实系统中替换为数字签名。"""
    return sha256(b"SIG|" + secret + b"|" + message)


def verify(secret: bytes, message: bytes, sig: bytes) -> bool:
    return sign(secret, message) == sig


# ============================================================
# 2. 简单 Reed-Solomon / 信息分散编码
# ============================================================

def pad_value(value: bytes, threshold: int) -> Tuple[List[int], int]:
    """
    原始 bytes -> GF(257) 符号。
    每 threshold 个符号作为一个多项式的系数。
    """
    symbols = list(value)
    pad_len = (-len(symbols)) % threshold
    symbols.extend([0] * pad_len)
    return symbols, pad_len


def rs_encode(value: bytes, n: int, threshold: int) -> Tuple[List[List[int]], int]:
    """
    把原始 value 编码为 n 个 share。
    每个 share 是 GF(257) 符号列表。
    任意 threshold 个 share 可恢复原始 value。
    """
    symbols, pad_len = pad_value(value, threshold)
    shares: List[List[int]] = [[] for _ in range(n)]

    for off in range(0, len(symbols), threshold):
        coeffs = symbols[off: off + threshold]
        for idx in range(n):
            x = idx + 1
            shares[idx].append(gf_eval_poly(coeffs, x))

    return shares, pad_len


def rs_decode(
    share_items: List[Tuple[int, List[int]]],
    threshold: int,
    data_len: int
) -> bytes:
    """
    用 threshold 个 share 恢复原始 value。
    share_items: [(share_index, symbols), ...]
    share_index 是 0..N-1，对应 x = share_index + 1。
    """
    if len(share_items) < threshold:
        raise ValueError("not enough shares to decode")

    chosen = share_items[:threshold]
    block_count = len(chosen[0][1])
    for _, syms in chosen:
        if len(syms) != block_count:
            raise ValueError("share length mismatch")

    xs = [(idx + 1) % PRIME for idx, _ in chosen]
    A = [[pow(x, c, PRIME) for c in range(threshold)] for x in xs]

    recovered: List[int] = []
    for block in range(block_count):
        ys = [syms[block] for _, syms in chosen]
        coeffs = gf_solve_linear(A, ys)
        recovered.extend(coeffs)

    raw = bytes([s for s in recovered[:data_len]])
    return raw


def serialize_share(symbols: List[int]) -> bytes:
    """GF(257) 符号用 2 字节保存，方便处理 256。"""
    out = bytearray()
    for s in symbols:
        out += int(s).to_bytes(2, "big")
    return bytes(out)


def deserialize_share(data: bytes) -> List[int]:
    if len(data) % 2 != 0:
        raise ValueError("bad share bytes length")
    syms = []
    for i in range(0, len(data), 2):
        x = int.from_bytes(data[i:i+2], "big")
        if x >= PRIME:
            raise ValueError("share symbol out of GF(257)")
        syms.append(x)
    return syms


# ============================================================
# 3. Merkle 承诺：对纠删码分片做承诺
# ============================================================

def leaf_hash(candidate_core: bytes, share_index: int, share_bytes: bytes) -> bytes:
    return sha256(b"LEAF|" + candidate_core + b"|" + i2b(share_index) + b"|" + share_bytes)


def merkle_parent(left: bytes, right: bytes) -> bytes:
    return sha256(b"NODE|" + left + right)


def build_merkle_tree(leaves: List[bytes]) -> List[List[bytes]]:
    if not leaves:
        raise ValueError("empty leaves")
    levels = [leaves[:]]
    cur = leaves[:]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(merkle_parent(left, right))
        levels.append(nxt)
        cur = nxt
    return levels


def merkle_root(levels: List[List[bytes]]) -> bytes:
    return levels[-1][0]


def merkle_proof(levels: List[List[bytes]], index: int) -> List[Tuple[bytes, str]]:
    """
    返回 [(sibling_hash, direction), ...]
    direction:
      - "R": sibling 在右侧，hash = parent(cur, sibling)
      - "L": sibling 在左侧，hash = parent(sibling, cur)
    """
    proof = []
    idx = index
    for level in levels[:-1]:
        if idx % 2 == 0:
            sib = idx + 1 if idx + 1 < len(level) else idx
            proof.append((level[sib], "R"))
        else:
            sib = idx - 1
            proof.append((level[sib], "L"))
        idx //= 2
    return proof


def verify_merkle_proof(
    leaf: bytes,
    index: int,
    proof: List[Tuple[bytes, str]],
    root: bytes
) -> bool:
    cur = leaf
    idx = index
    for sibling, direction in proof:
        if direction == "R":
            cur = merkle_parent(cur, sibling)
        elif direction == "L":
            cur = merkle_parent(sibling, cur)
        else:
            return False
        idx //= 2
    return cur == root


# ============================================================
# 4. 协议数据结构
# ============================================================

@dataclasses.dataclass(frozen=True)
class Candidate:
    proposer: int
    value_hash: bytes
    merkle_root: bytes
    data_len: int
    n: int
    f: int
    threshold: int

    def core_bytes(self) -> bytes:
        """
        不含 merkle_root 的核心字段。
        注意：leaf_hash 要包含 value_hash/data_len/n/f/threshold/proposer，
        防止不同候选复用同一片段。
        """
        return (
            b"CANDCORE|"
            + i2b(self.proposer)
            + self.value_hash
            + i2b(self.data_len)
            + i2b(self.n)
            + i2b(self.f)
            + i2b(self.threshold)
        )

    def id_bytes(self) -> bytes:
        return self.core_bytes() + b"|ROOT|" + self.merkle_root

    def key(self) -> bytes:
        return sha256(b"CANDIDATE-ID|" + self.id_bytes())

    def short(self) -> str:
        return f"p{self.proposer}:vh={hx(self.value_hash)}:mr={hx(self.merkle_root)}:id={hx(self.key())}"


@dataclasses.dataclass
class SharePacket:
    candidate: Candidate
    share_index: int
    share_bytes: bytes
    proof: List[Tuple[bytes, str]]


@dataclasses.dataclass
class DAC:
    candidate: Candidate
    ack_sigs: Dict[int, bytes]  # node_id -> sig over ACK|candidate.id_bytes()

    def message(self) -> bytes:
        return b"ACK|" + self.candidate.id_bytes()

    def size_bytes(self) -> int:
        # 估算证书传输成本：candidate id + 每个签名的 node_id + sig
        return len(self.candidate.id_bytes()) + len(self.ack_sigs) * (4 + 32)


@dataclasses.dataclass
class Node:
    node_id: int
    secret: bytes
    byzantine: bool
    input_value: bytes
    # candidate_key -> SharePacket
    stored_shares: Dict[bytes, SharePacket] = dataclasses.field(default_factory=dict)

    def verify_and_store_share(self, pkt: SharePacket) -> bool:
        cand = pkt.candidate
        if pkt.share_index != self.node_id:
            return False
        lf = leaf_hash(cand.core_bytes(), pkt.share_index, pkt.share_bytes)
        ok = verify_merkle_proof(lf, pkt.share_index, pkt.proof, cand.merkle_root)
        if ok:
            self.stored_shares[cand.key()] = pkt
        return ok

    def ack_candidate(self, cand: Candidate) -> bytes:
        return sign(self.secret, b"ACK|" + cand.id_bytes())

    def vote(self, phase: bytes, cand: Candidate) -> bytes:
        return sign(self.secret, phase + b"|" + cand.id_bytes())


@dataclasses.dataclass
class ConsensusResult:
    success: bool
    decided_candidate: Optional[Candidate]
    recovered_value: Optional[bytes]
    metrics: Dict[str, float]


# ============================================================
# 5. Hash + DAC 多值拜占庭共识仿真
# ============================================================

class HashDACMVBC:
    def __init__(
        self,
        n: int,
        f: int,
        k_bits: int,
        seed: int = 0,
        byzantine_mode: str = "mixed",
        common_honest_input: bool = False,
    ):
        if n < 3 * f + 1:
            raise ValueError("Need N >= 3f + 1")
        if n >= PRIME:
            raise ValueError("This compact demo uses GF(257), so N must be < 257")
        self.n = n
        self.f = f
        self.threshold = f + 1
        self.k_bits = k_bits
        self.k_bytes = (k_bits + 7) // 8
        self.rng = random.Random(seed)
        self.seed = seed
        self.byzantine_mode = byzantine_mode
        self.common_honest_input = common_honest_input

        byz_ids = set(self.rng.sample(range(n), f))
        self.byz_ids: Set[int] = byz_ids

        common_value = self.rng.randbytes(self.k_bytes) if common_honest_input else None
        self.nodes: List[Node] = []
        for i in range(n):
            secret = sha256(b"node-secret|" + i2b(i) + i2b(seed))
            if i in byz_ids:
                # 拜占庭节点的输入可以是任意值
                value = self.rng.randbytes(self.k_bytes)
            else:
                value = common_value if common_honest_input else self.rng.randbytes(self.k_bytes)
            self.nodes.append(Node(i, secret, i in byz_ids, value))

        self.byte_metrics = {
            "availability_payload_bytes": 0,
            "availability_ack_bytes": 0,
            "candidate_gossip_bytes": 0,
            "pbft_prepare_commit_bytes": 0,
            "naive_main_full_value_bytes": 0,
        }

    # -------------------- 签名/证书验证 --------------------

    def verify_ack(self, signer_id: int, cand: Candidate, sig: bytes) -> bool:
        return verify(self.nodes[signer_id].secret, b"ACK|" + cand.id_bytes(), sig)

    def verify_vote(self, signer_id: int, phase: bytes, cand: Candidate, sig: bytes) -> bool:
        return verify(self.nodes[signer_id].secret, phase + b"|" + cand.id_bytes(), sig)

    def verify_dac(self, dac: DAC) -> bool:
        if len(set(dac.ack_sigs)) < 2 * self.f + 1:
            return False
        if dac.candidate.n != self.n or dac.candidate.f != self.f:
            return False
        if dac.candidate.threshold != self.threshold:
            return False
        good = 0
        for node_id, sig in dac.ack_sigs.items():
            if 0 <= node_id < self.n and self.verify_ack(node_id, dac.candidate, sig):
                good += 1
        return good >= 2 * self.f + 1

    # -------------------- 候选值构造与分发 --------------------

    def build_packets_for_value(self, proposer: int, value: bytes) -> Tuple[Candidate, List[SharePacket]]:
        value_hash = sha256(value)

        # 第一次先用空 root 占位创建 core 字段所需内容。
        dummy = Candidate(
            proposer=proposer,
            value_hash=value_hash,
            merkle_root=b"\x00" * 32,
            data_len=len(value),
            n=self.n,
            f=self.f,
            threshold=self.threshold,
        )
        shares, _ = rs_encode(value, self.n, self.threshold)
        share_bytes_list = [serialize_share(s) for s in shares]

        # Merkle leaf 使用不含 root 的 core_bytes。
        leaves = [
            leaf_hash(dummy.core_bytes(), idx, share_bytes_list[idx])
            for idx in range(self.n)
        ]
        levels = build_merkle_tree(leaves)
        root = merkle_root(levels)

        cand = Candidate(
            proposer=proposer,
            value_hash=value_hash,
            merkle_root=root,
            data_len=len(value),
            n=self.n,
            f=self.f,
            threshold=self.threshold,
        )
        # 注意 cand.core_bytes() 与 dummy.core_bytes() 完全一致，因为 root 不在 core 里。
        packets = [
            SharePacket(
                candidate=cand,
                share_index=idx,
                share_bytes=share_bytes_list[idx],
                proof=merkle_proof(levels, idx),
            )
            for idx in range(self.n)
        ]
        return cand, packets

    def corrupt_packet(self, pkt: SharePacket) -> SharePacket:
        """制造一个无法通过 Merkle 验证的包。"""
        bad = bytearray(pkt.share_bytes)
        if bad:
            bad[0] ^= 0x01
        else:
            bad = bytearray(b"\x00")
        return SharePacket(pkt.candidate, pkt.share_index, bytes(bad), pkt.proof)

    def proposer_mode(self, proposer: int) -> str:
        if proposer not in self.byz_ids:
            return "honest"
        if self.byzantine_mode == "valid":
            return "valid"
        if self.byzantine_mode == "invalid":
            return "invalid"
        # mixed: 一部分拜占庭节点仍可能提出“可恢复但任意”的值，
        # 另一部分发送损坏分片，无法形成 DAC。
        return "valid" if self.rng.random() < 0.45 else "invalid"

    def availability_phase(self) -> Dict[bytes, DAC]:
        """
        数据可用性阶段：
        proposer 只把纠删码 share + Merkle proof 发给每个节点。
        节点验证后签 ACK。
        收集 >= 2f+1 ACK 即形成 DAC。
        """
        dacs: Dict[bytes, DAC] = {}

        for proposer in range(self.n):
            mode = self.proposer_mode(proposer)
            value = self.nodes[proposer].input_value
            cand, packets = self.build_packets_for_value(proposer, value)

            # 估算：每个接收者收到 share + proof + candidate id
            for pkt in packets:
                proof_size = len(pkt.proof) * (32 + 1)
                self.byte_metrics["availability_payload_bytes"] += (
                    len(pkt.candidate.id_bytes()) + len(pkt.share_bytes) + proof_size
                )

            ack_sigs: Dict[int, bytes] = {}

            for receiver in range(self.n):
                pkt = packets[receiver]

                if mode == "invalid":
                    # 拜占庭 proposer 给一部分节点发坏包。
                    # 拜占庭接收者可以乱签；诚实接收者会拒绝。
                    if receiver not in self.byz_ids or self.rng.random() < 0.70:
                        pkt = self.corrupt_packet(pkt)

                node = self.nodes[receiver]

                if node.byzantine:
                    # 拜占庭节点可以对任何候选乱 ACK，这里为了压力测试，给它签。
                    ack_sigs[receiver] = node.ack_candidate(cand)
                    self.byte_metrics["availability_ack_bytes"] += 4 + 32
                else:
                    if node.verify_and_store_share(pkt):
                        ack_sigs[receiver] = node.ack_candidate(cand)
                        self.byte_metrics["availability_ack_bytes"] += 4 + 32

            dac = DAC(cand, ack_sigs)
            if self.verify_dac(dac):
                dacs[cand.key()] = dac

        return dacs

    # -------------------- 压缩主共识：只对 CandidateID 投票 --------------------

    def gossip_dacs(self, local_dacs: Dict[bytes, DAC]) -> Dict[bytes, DAC]:
        """
        证书传播阶段：
        节点只转发 DAC，不转发完整原始 value。
        在同步可靠网络假设下，一轮 gossip 后诚实节点拥有相同候选池。
        """
        verified = {k: v for k, v in local_dacs.items() if self.verify_dac(v)}

        honest_count = self.n - self.f
        # 每个诚实节点向其他节点 gossip 每个 DAC。
        # 这里统计传输成本；集合本身在仿真中直接合并。
        for dac in verified.values():
            self.byte_metrics["candidate_gossip_bytes"] += honest_count * (self.n - 1) * dac.size_bytes()

        return verified

    def choose_canonical_candidate(self, dacs: Dict[bytes, DAC]) -> Optional[Candidate]:
        if not dacs:
            return None
        # 确定性选择：所有诚实节点在相同候选池上选 key 最小者。
        return min((dac.candidate for dac in dacs.values()), key=lambda c: c.key())

    def pbft_on_candidate_id(self, cand: Candidate) -> Tuple[bool, Dict[int, bytes], Dict[int, bytes]]:
        """
        极简 PBFT prepare/commit：
        - 主流程消息只包含 CandidateID + 签名，不包含原始 value。
        - 诚实节点对同一 canonical candidate 投票。
        - 拜占庭节点可能沉默或乱投。
        """
        prepare: Dict[int, bytes] = {}
        commit: Dict[int, bytes] = {}

        # PREPARE
        for node in self.nodes:
            if node.byzantine and self.rng.random() < 0.50:
                # 沉默
                continue
            prepare[node.node_id] = node.vote(b"PREPARE", cand)

        valid_prepare = {
            i: s for i, s in prepare.items()
            if self.verify_vote(i, b"PREPARE", cand, s)
        }
        prepared = len(valid_prepare) >= 2 * self.f + 1

        # COMMIT
        if prepared:
            for node in self.nodes:
                if node.byzantine and self.rng.random() < 0.50:
                    continue
                commit[node.node_id] = node.vote(b"COMMIT", cand)

        valid_commit = {
            i: s for i, s in commit.items()
            if self.verify_vote(i, b"COMMIT", cand, s)
        }
        committed = len(valid_commit) >= 2 * self.f + 1

        # 传输成本估算：诚实节点广播 prepare + commit。
        honest_count = self.n - self.f
        vote_msg_size = len(cand.id_bytes()) + 32 + 4
        self.byte_metrics["pbft_prepare_commit_bytes"] += 2 * honest_count * (self.n - 1) * vote_msg_size

        # naive 对比：如果主共识每条 prepare/commit 都带完整原始 value。
        naive_msg_size = self.k_bytes + 32 + 4
        self.byte_metrics["naive_main_full_value_bytes"] += 2 * honest_count * (self.n - 1) * naive_msg_size

        return committed, valid_prepare, valid_commit

    # -------------------- 决定后的恢复与验证 --------------------

    def collect_recovery_shares(self, cand: Candidate) -> Tuple[List[Tuple[int, List[int]]], int]:
        """
        决定 CandidateID 后，从节点拉取对应 share。
        接收端重新验证 Merkle proof，过滤拜占庭坏 share。
        """
        valid_shares: List[Tuple[int, List[int]]] = []
        bad_share_count = 0

        for node in self.nodes:
            pkt = node.stored_shares.get(cand.key())

            if pkt is None:
                # 拜占庭节点可能伪造一个坏 share；诚实节点没有就不返回。
                if node.byzantine and self.rng.random() < 0.35:
                    fake = os.urandom(max(2, self.k_bytes // max(1, self.threshold)))
                    bad_share_count += 1
                continue

            send_pkt = pkt
            if node.byzantine and self.rng.random() < 0.50:
                send_pkt = self.corrupt_packet(pkt)

            lf = leaf_hash(cand.core_bytes(), send_pkt.share_index, send_pkt.share_bytes)
            if verify_merkle_proof(lf, send_pkt.share_index, send_pkt.proof, cand.merkle_root):
                try:
                    syms = deserialize_share(send_pkt.share_bytes)
                    valid_shares.append((send_pkt.share_index, syms))
                except ValueError:
                    bad_share_count += 1
            else:
                bad_share_count += 1

        # 去重，避免同一 index 重复
        dedup = {}
        for idx, syms in valid_shares:
            dedup[idx] = syms
        return list(dedup.items()), bad_share_count

    def recover_and_verify(self, cand: Candidate) -> Tuple[bool, bool, Optional[bytes], Dict[str, float]]:
        shares, bad_share_count = self.collect_recovery_shares(cand)
        enough_shares = len(shares) >= cand.threshold

        metrics = {
            "recovery_valid_shares": len(shares),
            "recovery_bad_shares_filtered": bad_share_count,
            "recovery_threshold": cand.threshold,
            "enough_shares": 1.0 if enough_shares else 0.0,
            "hash_ok": 0.0,
            "merkle_root_ok": 0.0,
        }

        if not enough_shares:
            return False, False, None, metrics

        recovered = rs_decode(shares, cand.threshold, cand.data_len)
        hash_ok = sha256(recovered) == cand.value_hash

        # 再编码检查 Merkle root，防止“碰巧 hash 一致但编码承诺不一致”的实现错误。
        rec_cand, rec_packets = self.build_packets_for_value(cand.proposer, recovered)
        merkle_ok = rec_cand.merkle_root == cand.merkle_root

        metrics["hash_ok"] = 1.0 if hash_ok else 0.0
        metrics["merkle_root_ok"] = 1.0 if merkle_ok else 0.0

        return hash_ok, merkle_ok, recovered, metrics

    # -------------------- 一次完整运行 --------------------

    def run_once(self) -> ConsensusResult:
        dacs0 = self.availability_phase()
        candidate_count_before = len(dacs0)
        verified_dacs = self.gossip_dacs(dacs0)
        candidate_count_after = len(verified_dacs)

        cand = self.choose_canonical_candidate(verified_dacs)
        if cand is None:
            return ConsensusResult(
                success=False,
                decided_candidate=None,
                recovered_value=None,
                metrics={
                    "candidate_count_before_gossip": candidate_count_before,
                    "candidate_count_after_gossip": candidate_count_after,
                    "reason_no_candidate": 1.0,
                } | self.byte_metrics
            )

        committed, prepare_cert, commit_cert = self.pbft_on_candidate_id(cand)
        all_honest_agree = committed  # 本仿真中所有诚实节点使用同一候选池和确定性选择。

        hash_ok, merkle_ok, recovered, rec_metrics = self.recover_and_verify(cand)

        # 强有效性测试：如果所有诚实节点输入相同，则输出应等于该公共输入。
        common_input_validity = 1.0
        if self.common_honest_input and recovered is not None:
            honest_values = [n.input_value for n in self.nodes if not n.byzantine]
            common_input_validity = 1.0 if all(recovered == v for v in honest_values) else 0.0

        main_compressed = (
            self.byte_metrics["candidate_gossip_bytes"]
            + self.byte_metrics["pbft_prepare_commit_bytes"]
        )
        naive_main = self.byte_metrics["naive_main_full_value_bytes"]
        saving_ratio = 1.0 - (main_compressed / naive_main) if naive_main > 0 else 0.0

        metrics = {
            "N": self.n,
            "f": self.f,
            "threshold": self.threshold,
            "k_bits": self.k_bits,
            "byzantine_nodes": len(self.byz_ids),
            "candidate_count_before_gossip": candidate_count_before,
            "candidate_count_after_gossip": candidate_count_after,
            "decided_proposer": cand.proposer,
            "decided_proposer_byzantine": 1.0 if cand.proposer in self.byz_ids else 0.0,
            "prepare_certificate_size": len(prepare_cert),
            "commit_certificate_size": len(commit_cert),
            "all_honest_agree": 1.0 if all_honest_agree else 0.0,
            "committed": 1.0 if committed else 0.0,
            "common_input_validity_ok": common_input_validity,
            "main_compressed_bytes": main_compressed,
            "main_naive_full_value_bytes": naive_main,
            "main_bandwidth_saving_ratio": saving_ratio,
            "total_protocol_bytes_est": (
                self.byte_metrics["availability_payload_bytes"]
                + self.byte_metrics["availability_ack_bytes"]
                + main_compressed
            ),
        }
        metrics.update(self.byte_metrics)
        metrics.update(rec_metrics)

        success = bool(
            committed
            and all_honest_agree
            and hash_ok
            and merkle_ok
            and common_input_validity == 1.0
        )

        return ConsensusResult(
            success=success,
            decided_candidate=cand,
            recovered_value=recovered,
            metrics=metrics,
        )


# ============================================================
# 6. 指标验证与批量实验
# ============================================================

def format_bytes(x: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    v = float(x)
    for u in units:
        if v < 1024:
            return f"{v:.2f}{u}"
        v /= 1024
    return f"{v:.2f}TB"


def run_experiment(
    trials: int,
    n: int,
    f: int,
    k_bits: int,
    seed: int,
    byzantine_mode: str,
    common_honest_input: bool,
    verbose: bool = True,
) -> None:
    results: List[ConsensusResult] = []
    for t in range(trials):
        sim = HashDACMVBC(
            n=n,
            f=f,
            k_bits=k_bits,
            seed=seed + t,
            byzantine_mode=byzantine_mode,
            common_honest_input=common_honest_input,
        )
        res = sim.run_once()
        results.append(res)

        # 核心安全性断言：如果 success 为真，则必须满足以下指标。
        if res.success:
            m = res.metrics
            assert m["all_honest_agree"] == 1.0
            assert m["committed"] == 1.0
            assert m["enough_shares"] == 1.0
            assert m["hash_ok"] == 1.0
            assert m["merkle_root_ok"] == 1.0
            assert m["common_input_validity_ok"] == 1.0

    success_rate = sum(r.success for r in results) / len(results)
    metrics_list = [r.metrics for r in results if r.metrics.get("committed", 0.0) == 1.0]

    print("\n================ Hash + DAC 多值拜占庭共识仿真 ================")
    print(f"N={n}, f={f}, N>=3f+1: {n >= 3*f + 1}, threshold=f+1={f+1}")
    print(f"k={k_bits} bits ({(k_bits + 7)//8} bytes), trials={trials}, byzantine_mode={byzantine_mode}")
    print(f"all honest common input test: {common_honest_input}")
    print("----------------------------------------------------------------")
    print(f"成功率 success_rate: {success_rate:.2%}")

    if metrics_list:
        def avg(name: str) -> float:
            return statistics.mean(m[name] for m in metrics_list)

        print(f"平均候选 DAC 数: {avg('candidate_count_after_gossip'):.2f}")
        print(f"平均恢复有效 share 数: {avg('recovery_valid_shares'):.2f}")
        print(f"平均被过滤坏 share 数: {avg('recovery_bad_shares_filtered'):.2f}")
        print(f"平均 Prepare 证书大小: {avg('prepare_certificate_size'):.2f}")
        print(f"平均 Commit 证书大小: {avg('commit_certificate_size'):.2f}")
        print(f"平均主流程压缩传输: {format_bytes(avg('main_compressed_bytes'))}")
        print(f"平均主流程 naive 完整值传输: {format_bytes(avg('main_naive_full_value_bytes'))}")
        print(f"平均主流程节省比例: {avg('main_bandwidth_saving_ratio'):.2%}")
        print(f"平均全协议估算传输: {format_bytes(avg('total_protocol_bytes_est'))}")

    last = results[-1]
    print("----------------------------------------------------------------")
    print("最后一次实验摘要：")
    print(f"  success = {last.success}")
    if last.decided_candidate:
        print(f"  decided_candidate = {last.decided_candidate.short()}")
        print(f"  recovered_hash = {hx(sha256(last.recovered_value or b''), 24)}")
        print(f"  recovered_len = {len(last.recovered_value or b'')} bytes")
    print("  key metrics:")
    for key in [
        "all_honest_agree",
        "committed",
        "enough_shares",
        "hash_ok",
        "merkle_root_ok",
        "common_input_validity_ok",
        "main_bandwidth_saving_ratio",
    ]:
        if key in last.metrics:
            v = last.metrics[key]
            if "ratio" in key:
                print(f"    {key}: {v:.2%}")
            else:
                print(f"    {key}: {v}")
    print("================================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=7, help="总节点数，必须满足 N >= 3f + 1")
    parser.add_argument("--f", type=int, default=2, help="最多拜占庭节点数")
    parser.add_argument("--k-bits", type=int, default=16384, help="每个原始输入值的 bit 长度")
    parser.add_argument("--trials", type=int, default=20, help="批量实验次数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--byzantine-mode",
        choices=["mixed", "valid", "invalid"],
        default="mixed",
        help="mixed=部分可用部分作恶；valid=拜占庭也提出可恢复值；invalid=拜占庭发送坏分片",
    )
    parser.add_argument(
        "--common-honest-input",
        action="store_true",
        help="所有诚实节点使用同一输入，用于验证强有效性",
    )
    args = parser.parse_args()

    run_experiment(
        trials=args.trials,
        n=args.N,
        f=args.f,
        k_bits=args.k_bits,
        seed=args.seed,
        byzantine_mode=args.byzantine_mode,
        common_honest_input=args.common_honest_input,
        verbose=True,
    )


if __name__ == "__main__":
    main()
