# 自然语言处理词义替换工具

## 项目简介

基于 CEFR 语言等级的英文词义替换系统。输入句子 + 源 CEFR 等级 + 目标 CEFR 等级，自动替换词汇为对应难度的近义词，用于语言教学中的文本难度自适应调整。

## 代码架构

### 主入口与测试框架 (`main.py`)

```python
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

def load_unit_tests(path="unit_tests.csv"):
    """加载公开测试用例
    必填列: sentence, source_level, target_level
    可选列: expected_output"""

def validate_levels(source_level, target_level):
    """验证 CEFR 等级 ∈ [A1,A2,B1,B2,C1,C2]"""

def load_student_module(zid_name):
    """动态 import 学生模块
    模块必须定义: transform_sentence(sentence, source_level, target_level)"""
```

### 核心替换逻辑 (`Z5519231.py`)

```python
def transform_sentence(sentence: str, source_level: str, target_level: str) -> str:
    """
    1. spaCy 分词 + 词性标注 (POS tagging)
       - Token.pos_: NOUN/VERB/ADJ/ADV (实词)
       - Token.lemma_: 词元化 (如 running → run)

    2. 对每个实词:
       a. NLTK WordNet 查找同义词集 (Synsets)
          - wn.synsets(lemma, pos=wordnet_pos)
       b. CEFR 等级过滤:
          - 提升难度 (如 A2→B2): 选择更长/更少见的词
          - 降低难度 (如 B2→A1): 选择常用基础词
       c. pyinflect 词形变换:
          - 动词: 时态 (past/present), 人称 (3rd singular)
          - 名词: 单复数
          - 形容词/副词: 比较级/最高级

    3. 重组句子: 替换词 + 保持原始标点与句子结构
    """
```

### 测试 (`test.py`)

```python
# 单元测试: 等级映射、词汇替换正确性、边缘情况
# 边界测试: 空句子、未知词 (OOV)、极端等级跨级 (如 C2→A1)
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 分词/词性标注 | spaCy (en_core_web_sm, token.pos_, token.lemma_) |
| 同义词库 | NLTK WordNet (synsets, lemmas, hypernyms) |
| 词形变换 | pyinflect (动词时态/人称, 名词单复数, 形容词级) |
| 难度分级 | CEFR (A1-C2, 六等级) |
| 测试 | unit_tests.csv + test.py 自动化验证 |

## 运行方式

```bash
pip install spacy nltk pandas pyinflect
python -m spacy download en_core_web_sm
python main.py   # 加载测试用例并验证
python test.py   # 单元测试
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `main.py` | 主入口 + 测试框架 + 模块加载 |
| `Z5519231.py` | transform_sentence 核心实现 |
| `test.py` | 单元测试与边界情况 |
| `requirements.txt` | 依赖清单 |
