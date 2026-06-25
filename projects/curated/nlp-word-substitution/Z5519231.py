import math
from pathlib import Path
from collections import Counter
import pandas as pd
import spacy
import pyinflect
from nltk.corpus import wordnet
from nltk.wsd import lesk

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Simple list and dict for CEFR levels
cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
level_map = {}
for i, level in enumerate(cefr_levels):
    level_map[level] = i + 1


def load_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")

    df = pd.read_csv(file_path)
    if "text" not in df.columns or "cefr_level" not in df.columns:
        raise ValueError("data.csv must contain columns: text, cefr_level")

    return df


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli

    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Global variables for the language model
word_levels = {}
unigram_counts = Counter()
bigram_counts = Counter()
vocab_size = 0
total_words = 0


def init_model():
    """Builds the N-gram models and finds CEFR level for each word."""
    global word_levels, unigram_counts, bigram_counts, vocab_size, total_words

    try:
        df = load_data("data.csv")
    except Exception:
        return

    level_word_counts = {}
    level_total_words = {}
    for L in range(1, 7):
        level_word_counts[L] = Counter()
        level_total_words[L] = 0

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Loading Data", ncols=80):
        text = str(row["text"])
        level_str = row["cefr_level"]

        if level_str not in level_map:
            continue

        level_int = level_map[level_str]
        doc = nlp(text)

        prev_word = "<s>"
        unigram_counts["<s>"] += 1

        for token in doc:
            lemma = token.lemma_.lower()

            unigram_counts[lemma] += 1
            bigram_counts[(prev_word, lemma)] += 1
            prev_word = lemma

            # Count only real words for difficulty level
            if token.is_alpha and not token.is_stop:
                level_word_counts[level_int][lemma] += 1
                level_total_words[level_int] += 1

        bigram_counts[(prev_word, "</s>")] += 1
        unigram_counts["</s>"] += 1

    temp_dict = {}
    for lemma in unigram_counts:
        # Calculate prob of this word in each level
        probs = {}
        for L in range(1, 7):
            if level_total_words[L] > 0:
                probs[L] = level_word_counts[L][lemma] / level_total_words[L]
            else:
                probs[L] = 0

        total_prob = sum(probs.values())
        if total_prob == 0:
            temp_dict[lemma] = 1
            continue

        # Find the 25% threshold to assign level
        current_sum = 0.0
        assigned_level = 6
        for L in range(1, 7):
            normalized_prob = probs[L] / total_prob
            current_sum += normalized_prob
            if current_sum >= 0.25:
                assigned_level = L
                break

        temp_dict[lemma] = assigned_level

    word_levels = temp_dict
    vocab_size = len(unigram_counts)
    total_words = sum(unigram_counts.values())


# run this when imported
init_model()


def get_wn_pos(spacy_tag):
    """Convert Spacy POS tags to WordNet format."""
    if spacy_tag.startswith('J'):
        return wordnet.ADJ
    elif spacy_tag.startswith('V'):
        return wordnet.VERB
    elif spacy_tag.startswith('N'):
        return wordnet.NOUN
    elif spacy_tag.startswith('R'):
        return wordnet.ADV
    return None


def get_lm_score(prev_w, curr_w, next_w):
    """Calculate bigram score with smoothing."""
    if vocab_size > 0:
        p_curr = (unigram_counts.get(curr_w, 0) + 1) / (total_words + vocab_size)
        p_next = (unigram_counts.get(next_w, 0) + 1) / (total_words + vocab_size)
    else:
        p_curr = 1e-5
        p_next = 1e-5

    # Forward bigram
    count_fwd = bigram_counts.get((prev_w, curr_w), 0)
    count_prev = unigram_counts.get(prev_w, 0)
    if count_prev > 0:
        p_fwd = count_fwd / count_prev
    else:
        p_fwd = 0.0

    # Backward bigram
    count_bwd = bigram_counts.get((curr_w, next_w), 0)
    count_curr = unigram_counts.get(curr_w, 0)
    if count_curr > 0:
        p_bwd = count_bwd / count_curr
    else:
        p_bwd = 0.0

    # Interpolation
    weight = 0.80
    score_fwd = weight * p_fwd + (1 - weight) * p_curr
    score_bwd = weight * p_bwd + (1 - weight) * p_next

    return math.log(score_fwd) + math.log(score_bwd)


def check_replace(lemma, source_int, target_int):
    """Check if we need to replace this word."""
    if lemma not in word_levels:
        return False

    current_lvl = word_levels[lemma]
    if source_int > target_int:
        return current_lvl > target_int
    elif source_int < target_int:
        return current_lvl < target_int
    return False


def transform_sentence(sentence, source_level, target_level):
    """
    Transforms a sentence from source CEFR level to target CEFR level.
    """
    if source_level not in cefr_levels or target_level not in cefr_levels:
        raise ValueError("Invalid CEFR level")

    if source_level == target_level:
        return sentence

    source_int = level_map[source_level]
    target_int = level_map[target_level]

    doc = nlp(sentence)
    result_tokens = []

    context_list = []
    for t in doc:
        context_list.append(t.text)

    for i, token in enumerate(doc):
        lemma = token.lemma_.lower()

        # Skip stop words, numbers and entities
        if not token.is_alpha or token.is_stop or token.ent_type_:
            result_tokens.append([token.text, token.whitespace_])
            continue

        if not check_replace(lemma, source_int, target_int):
            result_tokens.append([token.text, token.whitespace_])
            continue

        wn_pos = get_wn_pos(token.tag_)
        all_syns = wordnet.synsets(lemma, pos=wn_pos)
        if not all_syns:
            all_syns = wordnet.synsets(lemma)

        # 1. WSD using Lesk
        search_syns = set()
        best_syn = lesk(context_list, token.text, pos=wn_pos)

        if best_syn:
            search_syns.add(best_syn)
        if all_syns:
            search_syns.add(all_syns[0])

        # 2. Expand search space
        expanded_syns = set(search_syns)

        if source_int > target_int:
            # Look for simpler words
            for syn in list(expanded_syns):
                expanded_syns.update(syn.hypernyms())
                for h in syn.hypernyms():
                    expanded_syns.update(h.hypernyms())
                expanded_syns.update(syn.similar_tos())
        else:
            # Look for harder words
            for syn in list(expanded_syns):
                expanded_syns.update(syn.hyponyms())
                for h in syn.hyponyms():
                    expanded_syns.update(h.hyponyms())
                expanded_syns.update(syn.similar_tos())

        raw_cands = {}
        # 3. Filter by similarity to avoid wrong meanings
        for syn in expanded_syns:
            sim = 0.0
            if search_syns:
                sim_list = []
                for orig in search_syns:
                    s = orig.path_similarity(syn)
                    if s is not None:
                        sim_list.append(s)
                if sim_list:
                    sim = max(sim_list)

            # Drop words that are too different
            if sim < 0.16:
                continue

            for l in syn.lemmas():
                cand_name = l.name().lower().replace("_", " ")
                # No multi-word phrases
                if cand_name == lemma or " " in cand_name:
                    continue

                curr_max = raw_cands.get(cand_name, 0.0)
                raw_cands[cand_name] = max(curr_max, sim)

        valid_cands = {}
        original_lvl = word_levels.get(lemma, 6)

        # 4. Filter by level
        for cand, sim in raw_cands.items():
            if cand not in word_levels:
                # Guess level by length
                length = len(cand)
                if length <= 4:
                    cand_lvl = 1
                elif length <= 6:
                    cand_lvl = 2
                elif length <= 8:
                    cand_lvl = 3
                elif length <= 10:
                    cand_lvl = 4
                else:
                    cand_lvl = 5
            else:
                cand_lvl = word_levels[cand]

            freq_cand = unigram_counts.get(cand, 0)
            freq_orig = unigram_counts.get(lemma, 0)
            is_more_freq = freq_cand > freq_orig

            if source_int > target_int:
                if (cand_lvl <= target_int or cand_lvl < original_lvl) and is_more_freq:
                    valid_cands[cand] = sim
            else:
                if (cand_lvl >= target_int or cand_lvl > original_lvl) and not is_more_freq:
                    valid_cands[cand] = sim

        # Fallback if nothing is valid
        if not valid_cands:
            for cand, sim in raw_cands.items():
                if unigram_counts.get(cand, 0) > unigram_counts.get(lemma, 0):
                    if source_int > target_int and len(cand) <= len(lemma):
                        valid_cands[cand] = sim
                    elif source_int < target_int and len(cand) >= len(lemma):
                        valid_cands[cand] = sim

        if not valid_cands:
            result_tokens.append([token.text, token.whitespace_])
            continue

        # 5. Score using Bigram LM
        if i > 0:
            prev_lemma = doc[i - 1].lemma_.lower()
        else:
            prev_lemma = "<s>"

        if i < len(doc) - 1:
            next_lemma = doc[i + 1].lemma_.lower()
        else:
            next_lemma = "</s>"

        best_cand = None
        best_score = float('-inf')

        for cand, sim in valid_cands.items():
            lm_score = get_lm_score(prev_lemma, cand, next_lemma)
            total_score = lm_score + math.log(sim + 0.01)

            if total_score > best_score:
                best_score = total_score
                best_cand = cand

        # 6. Fix grammar
        final_word = best_cand
        if best_cand:
            inflections = pyinflect.getInflection(best_cand, token.tag_)
            if inflections:
                final_word = inflections[0]

        if token.text.istitle():
            final_word = final_word.title()
        elif token.text.isupper():
            final_word = final_word.upper()

        result_tokens.append([final_word, token.whitespace_])

    # 7. Fix a/an articles
    vowels = ('a', 'e', 'i', 'o', 'u')
    special_consonants = ('hour', 'honest', 'honor', 'heir')
    special_vowels = ('uni', 'use', 'eur', 'one', 'once')

    for idx in range(len(result_tokens) - 1):
        curr_w, curr_space = result_tokens[idx]

        if curr_w.lower() in ['a', 'an']:
            next_w = result_tokens[idx + 1][0].lower()

            if next_w:
                is_vowel_sound = next_w.startswith(vowels)

                if next_w.startswith(special_consonants):
                    is_vowel_sound = True
                elif next_w.startswith(special_vowels):
                    is_vowel_sound = False

                if is_vowel_sound and curr_w.lower() == 'a':
                    result_tokens[idx][0] = 'An' if curr_w.istitle() else 'an'
                elif not is_vowel_sound and curr_w.lower() == 'an':
                    result_tokens[idx][0] = 'A' if curr_w.istitle() else 'a'

    # Build final sentence
    final_text = ""
    for w, space in result_tokens:
        final_text += w + space

    return final_text.strip()