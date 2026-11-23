# Abbreviation Detection Using CRF

An NLP final project for **NYU Special Topics: Natural Language Processing (Fall 2026)**.

## Overview

This project detects abbreviations and their long forms in computer science text. Every
token gets one label (Zilio et al., 2024): `B-AC` (abbreviation), `B-LF` / `I-LF` (long
form), `B-O` (other). Systems are scored on abbreviation spans, long-form spans, and
(abbreviation, long form) pairs with exact-match precision, recall and F1.

## Project Structure

```text
CS-Acronym-NLP/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/SDU-train-data.json
│   ├── processed/SDU-train-data.partial.jsonl   # output of prepare_data.py
│   └── lexicon/english_words.txt                # English word list (SCOWL) for dictionary features
├── src/
│   ├── prepare_data.py   # SDU json -> tokens + BIO labels (jsonl)
│   ├── utils.py          # loading, train/dev split, BIO<->spans, S&H matcher, pair linking
│   ├── evaluate.py       # span-level P/R/F1 for abbreviations, long forms, pairs
│   ├── baseline.py       # majority-class and rule-based detector baselines
│   ├── rule_based.py     # Technique 1: Schwartz & Hearst (2003) + document propagation
│   ├── features.py       # Technique 2: CRF feature groups
│   ├── crf_model.py      # CRF wrapper (python-crfsuite, or numpy fallback)
│   ├── simple_crf.py     # numpy/scipy linear-chain CRF (fallback backend)
│   ├── pos.py            # optional spaCy POS tags
│   ├── train_crf.py      # train / evaluate / ablation / hyper-parameter search
│   └── predict.py        # run a model on raw text and list abbreviations + meanings
├── models/               # saved CRF (created by train_crf.py, git-ignored)
└── results/              # saved predictions / ablation tables (git-ignored)
```

## Setup

```bash
pip install -r requirements.txt
# optional, enables the CRF "pos" feature group:
pip install spacy && python -m spacy download en_core_web_sm
```

If `python-crfsuite` is not installed, the CRF still trains with the numpy/scipy
fallback in `simple_crf.py` (L2 only, a few minutes instead of seconds).

## Usage

```bash
python src/prepare_data.py                    # 1. build data/processed/*.jsonl

python src/baseline.py                        # 2. baselines on the dev split
python src/rule_based.py                      # 3. Schwartz & Hearst + propagation
python src/rule_based.py --compare            #    all rule-based variants
python src/train_crf.py                       # 4. train CRF, score dev, save models/crf.pkl
python src/train_crf.py --with-rules          #    + rule-based output as features
python src/train_crf.py --ablation            #    remove one feature group at a time
python src/train_crf.py --search              #    small c1/c2 grid on dev

python src/predict.py --text "We use a convolutional neural network (CNN). The CNN ..."
python src/predict.py --file paper.txt --model rules
```

Useful flags: `--save-preds` (writes `results/*.jsonl`, scoreable with
`python src/evaluate.py --gold dev --pred results/<file>.jsonl`) and `--errors out.txt`
(dumps dev mistakes for error analysis).

**Final test run** (once our annotated test set exists, after running it through
`prepare_data.py`):

```bash
python src/rule_based.py --split data/processed/<test>.jsonl
python src/train_crf.py --final --test data/processed/<test>.jsonl
```

## Preprocessing

`prepare_data.py` tokenizes with character offsets, converts annotations to BIO labels,
validates/repairs spans and exports JSONL.

| Metric | Count |
|---|---:|
| Original records | 3,980 |
| Successfully processed | 3,696 |
| Rejected | 284 |

## Data split

SDU sentences carry no paper id, so the 80/20 train/dev split (`utils.split_records`,
seed 13) is by sentence: 2,956 train / 740 dev. If a record has a `doc_id` field, the
split groups by it and propagation works across all sentences of that paper. Our own
test set should keep a `doc_id` per sentence for this (`prepare_data.py` currently drops
extra fields, so it needs to pass `doc_id` through).

## Models

**Baselines.** Majority class (all `B-O`), and a heuristic detector: 2–10 characters with
≥2 capitals, a short token alone in parentheses after a matching phrase, or an
abbreviation-like token missing from the English word list; with filters for all-caps
titles, name initials, Roman numerals (I/V/X only, so `CC`, `CD` survive) and single letters.

**Technique 1: rule-based (`rule_based.py`).** Schwartz & Hearst on tokens: candidates
from `LF ( SF )` / `SF ( LF )`, short-form filter, window of min(|SF|+5, 2|SF|) words,
right-to-left character matching, rejection rules. Extension: document-level propagation
of defined short forms (and plurals) to later uses. Optional extras: `--glossary`
(`SF = long form`, ~12% of SDU long forms) and `--with-detector` (undefined abbreviations).

**Technique 2: CRF (`train_crf.py`).** Joint labelling of abbreviations and long forms.
Feature groups (each removable for ablation): `form`, `ortho`, `context` (±2 tokens),
`paren`, `pos`, `dict`, `defmatch`, plus the optional `rules` group (S&H output as features).

## Status

Preprocessing, baselines, rule-based system and CRF implemented. Pending: final
hyper-parameters with `python-crfsuite`, feature ablation, test-set annotation and
the final evaluation.
