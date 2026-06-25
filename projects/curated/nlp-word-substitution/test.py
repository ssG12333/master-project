# Rename this file to your zID, e.g. z1234567.py

from pathlib import Path
import pandas as pd
import spacy
import pyinflect
import nltk
from nltk.corpus import wordnet
from nltk.wsd import lesk
from collections import defaultdict, Counter
import math

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_MAP = {level: i + 1 for i, level in enumerate(CEFR_LEVELS)}


# -------------------------------------------------
# Data & Model Initialization
# -------------------------------------------------

def load_training_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")
    df = pd.read_csv(file_path)
    if not {"text", "cefr_level"}.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli

    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

lexicon = {}
unigram_counts = Counter()
bigram_counts = Counter()
vocab_size = 0
total_tokens = 0


def build_models():
    """
    回归 数学期望定级法 (Expected Value Calibration)
    精准暴露复杂词汇！
    """
    global lexicon, unigram_counts, bigram_counts, vocab_size, total_tokens
    print(f"\n[模型构建] 开始读取数据并利用数学期望进行精准定级...")

    try:
        df = load_training_data("data.csv")
    except (FileNotFoundError, ValueError):
        return

    level_counts = {L: Counter() for L in range(1, 7)}
    level_totals = {L: 0 for L in range(1, 7)}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="训练 LM", ncols=80):
        text = str(row["text"])
        level_str = row["cefr_level"]
        if level_str not in LEVEL_MAP: continue

        level_int = LEVEL_MAP[level_str]
        doc = nlp(text)

        prev_lemma = "<s>"
        unigram_counts["<s>"] += 1

        for token in doc:
            lemma = token.lemma_.lower()
            unigram_counts[lemma] += 1
            bigram_counts[(prev_lemma, lemma)] += 1
            prev_lemma = lemma

            if token.is_alpha and not token.is_stop:
                level_counts[level_int][lemma] += 1
                level_totals[level_int] += 1

        bigram_counts[(prev_lemma, "</s>")] += 1
        unigram_counts["</s>"] += 1

    sorted_unigrams = unigram_counts.most_common()
    freq_ranks = {word: rank for rank, (word, count) in enumerate(sorted_unigrams)}

    temp_lexicon = {}
    for lemma in unigram_counts:
        p_word_given_level = {
            L: (level_counts[L][lemma] / level_totals[L]) if level_totals[L] > 0 else 0
            for L in range(1, 7)
        }

        sum_p = sum(p_word_given_level.values())
        if sum_p == 0:
            temp_lexicon[lemma] = 1
            continue

        norm_dist = {L: p_word_given_level[L] / sum_p for L in range(1, 7)}

        # 数学期望 (Mean) 定级法：能够精确计算出单词在各个难度下的平均分布
        expected_level = sum(L * norm_dist[L] for L in range(1, 7))
        assigned_level = round(expected_level)

        # 齐普夫基石：保护真正的高频基础词汇不被误判
        rank = freq_ranks.get(lemma, 999999)
        if rank < 1000:
            assigned_level = 1
        elif rank < 3000:
            assigned_level = min(assigned_level, 2)
        elif rank < 5000:
            assigned_level = min(assigned_level, 3)

        temp_lexicon[lemma] = assigned_level

    lexicon = temp_lexicon
    vocab_size = len(unigram_counts)
    total_tokens = sum(unigram_counts.values())
    print(f"[模型构建] 完毕。词汇库大小: {vocab_size}")


build_models()


def get_wordnet_pos(spacy_pos):
    if spacy_pos.startswith('J'):
        return wordnet.ADJ
    elif spacy_pos.startswith('V'):
        return wordnet.VERB
    elif spacy_pos.startswith('N'):
        return wordnet.NOUN
    elif spacy_pos.startswith('R'):
        return wordnet.ADV
    return None


def get_interpolated_prob(prev_word, current_word, next_word):
    """
    标准的线性插值 LM
    """
    p_uni_curr = (unigram_counts.get(current_word, 0) + 1) / (total_tokens + vocab_size) if vocab_size > 0 else 1e-5
    p_uni_next = (unigram_counts.get(next_word, 0) + 1) / (total_tokens + vocab_size) if vocab_size > 0 else 1e-5

    bg_fwd = bigram_counts.get((prev_word, current_word), 0)
    ug_prev = unigram_counts.get(prev_word, 0)
    p_fwd = bg_fwd / ug_prev if ug_prev > 0 else 0.0

    bg_bwd = bigram_counts.get((current_word, next_word), 0)
    ug_curr = unigram_counts.get(current_word, 0)
    p_bwd = bg_bwd / ug_curr if ug_curr > 0 else 0.0

    lmbda = 0.85
    p_int_fwd = lmbda * p_fwd + (1 - lmbda) * p_uni_curr
    p_int_bwd = lmbda * p_bwd + (1 - lmbda) * p_uni_next

    return math.log(p_int_fwd) + math.log(p_int_bwd)


def needs_replacement(lemma, source_int, target_int):
    if lemma not in lexicon: return False
    word_level = lexicon[lemma]
    if source_int > target_int:
        needs = word_level > target_int
        if needs: print(f"\n  >> [检测] '{lemma}' (定级 {word_level}) > 目标 {target_int}，需简化。")
        return needs
    elif source_int < target_int:
        needs = word_level < target_int
        if needs: print(f"\n  >> [检测] '{lemma}' (定级 {word_level}) < 目标 {target_int}，需提升。")
        return needs
    return False


# -------------------------------------------------
# 核心流水线
# -------------------------------------------------

def transform_sentence(sentence, source_level, target_level):
    print(f"\n{'=' * 60}")
    print(f"原句 [{source_level}]: {sentence}")
    print(f"{'-' * 60}")

    if source_level == target_level: return sentence

    source_int = LEVEL_MAP[source_level]
    target_int = LEVEL_MAP[target_level]

    doc = nlp(sentence)
    transformed_text = ""
    context_words = [t.text for t in doc]

    for i, token in enumerate(doc):
        lemma = token.lemma_.lower()

        if not token.is_alpha or token.is_stop or token.ent_type_:
            transformed_text += token.text_with_ws
            continue

        if not needs_replacement(lemma, source_int, target_int):
            transformed_text += token.text_with_ws
            continue

        wn_pos = get_wordnet_pos(token.tag_)

        # 完美解决 realise (英式) -> realize (美式) 识别不到的经典大坑
        all_synsets = wordnet.synsets(lemma, pos=wn_pos)
        if not all_synsets:
            all_synsets = wordnet.synsets(lemma)
        if not all_synsets and lemma.endswith('ise'):
            us_lemma = lemma[:-3] + 'ize'
            all_synsets = wordnet.synsets(us_lemma, pos=wn_pos) or wordnet.synsets(us_lemma)

        synsets_to_explore = set()
        best_synset = lesk(context_words, token.text, pos=wn_pos)
        if best_synset: synsets_to_explore.add(best_synset)
        if all_synsets: synsets_to_explore.add(all_synsets[0])

        expanded_synsets = set(synsets_to_explore)
        if source_int > target_int:
            for syn in list(expanded_synsets):
                expanded_synsets.update(syn.hypernyms())
                for h in syn.hypernyms():
                    expanded_synsets.update(h.hypernyms())
                expanded_synsets.update(syn.similar_tos())
        elif source_int < target_int:
            for syn in list(expanded_synsets):
                expanded_synsets.update(syn.hyponyms())
                for h in syn.hyponyms():
                    expanded_synsets.update(h.hyponyms())
                expanded_synsets.update(syn.similar_tos())

        raw_candidates = {}
        for syn in expanded_synsets:
            sim_score = 0.0
            if synsets_to_explore:
                sims = [orig.path_similarity(syn) for orig in synsets_to_explore if orig.path_similarity(syn) is not None]
                if sims: sim_score = max(sims)

            # 放宽准入条件至 0.14，允许更多词进入候选，交给下面严格的数学惩罚去处理
            if sim_score < 0.14: continue

            for l in syn.lemmas():
                cand_name = l.name().lower().replace("_", " ")
                if cand_name == lemma or " " in cand_name: continue
                raw_candidates[cand_name] = max(raw_candidates.get(cand_name, 0.0), sim_score)

        valid_candidates = {}
        word_level = lexicon.get(lemma, 6)

        for cand, sim in raw_candidates.items():
            if cand not in lexicon:
                cand_len = len(cand)
                cand_level = 1 if cand_len <= 4 else 2 if cand_len <= 6 else 3 if cand_len <= 8 else 4
            else:
                cand_level = lexicon[cand]

            is_more_frequent = unigram_counts.get(cand, 0) > unigram_counts.get(lemma, 0)

            if source_int > target_int:
                if (cand_level <= target_int or cand_level < word_level) and is_more_frequent:
                    valid_candidates[cand] = sim
            else:
                if (cand_level >= target_int or cand_level > word_level) and not is_more_frequent:
                    valid_candidates[cand] = sim

        # 如果没有符合频率和等级的，启用备用降级方案
        if not valid_candidates:
            for cand, sim in raw_candidates.items():
                if unigram_counts.get(cand, 0) > unigram_counts.get(lemma, 0):
                    if (source_int > target_int and len(cand) <= len(lemma)) or \
                            (source_int < target_int and len(cand) >= len(lemma)):
                        valid_candidates[cand] = sim

        if not valid_candidates:
            print(f"    [失败] 未找到合适的替换词。")
            transformed_text += token.text_with_ws
            continue

        best_candidate = None
        best_score = float('-inf')

        print(f"    [评分明细] 候选词对比 (原词: {lemma}):")
        for cand, sim in valid_candidates.items():
            lm_score = get_interpolated_prob(prev_lemma, cand, next_lemma)

            # 【核心微调】：给予极其严厉的语义漂移惩罚！
            # 相似度越低，扣分越狠！
            semantic_score = 20.0 * math.log(sim + 0.001)
            total_score = lm_score + semantic_score

            print(f"      -> {cand:<12} | 相似度: {sim:.2f} | 语义扣分: {semantic_score:>7.2f} | 语境得分(LM): {lm_score:>7.2f} | 总分: {total_score:>7.2f}")

            if total_score > best_score:
                best_score = total_score
                best_candidate = cand

        final_word = best_candidate
        if best_candidate:
            inflections = pyinflect.getInflection(best_candidate, token.tag_)
            if inflections:
                final_word = inflections[0]

        if token.text.istitle():
            final_word = final_word.title()
        elif token.text.isupper():
            final_word = final_word.upper()

        transformed_text += final_word + token.whitespace_
        print(f"    [胜出] ===> 成功替换为 '{final_word}'")

    final_sentence = transformed_text.strip()
    print(f"目标 [{target_level}]: {final_sentence}")
    return final_sentence


if __name__ == "__main__":
    test_cases = [
        ("I purchased a magnificent house yesterday.", "C1", "A2"),
        ("The children were extremely delighted with the performance.", "B2", "A2"),
        ("She attempted to resolve the complicated problem.", "B2", "B1"),
        ("The professor explained the concept very clearly.", "B1", "A2"),
        ("They constructed a large building near the river.", "B2", "A2"),
        ("He quickly realised his mistake.", "B2", "A2"),
        ("The committee will evaluate the proposal tomorrow.", "C1", "B1"),
        ("She was exhausted after completing the assignment.", "B2", "A2"),
        ("The results demonstrate a significant improvement.", "C1", "B1"),
        ("The scientist conducted an experiment.", "B2", "A2")
    ]

    print("\n========== 开始微调测试 ==========\n")
    for text, src, tgt in test_cases:
        transform_sentence(text, src, tgt)