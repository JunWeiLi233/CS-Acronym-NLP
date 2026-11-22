# Abbreviation Detection Using CRF

An NLP final project for **NYU Special Topics: Natural Language Processing (Fall 2026)**.

## Overview

This project aims to detect abbreviations and their long forms using a Conditional Random Field (CRF) model trained on the SDU abbreviation dataset.

## Project Structure

```text
CS-Acronym-NLP/
├── README.md
├── proposal/
│   └── final_project_proposal.pdf
├── data/
│   ├── raw/
│   │   └── SDU-train-data.json
│   └── processed/
│       ├── SDU-train-data.partial.jsonl
│       └── SDU-train-data.errors.json
├── src/
│   ├── prepare_data.py
│   ├── features.py
│   ├── train_crf.py
│   ├── baseline.py
│   ├── evaluate.py
│   └── predict.py
└── results/
```

*This structure includes planned files that have not yet been implemented.*

## Current Progress

The preprocessing pipeline (`prepare_data.py`) is complete.

It performs:
- Tokenization and character offset tracking
- Conversion of annotations into BIO labels
- Annotation validation and error correction
- Export to JSONL format

### Preprocessing Results

| Metric | Count |
|---|---:|
| Original records | 3,980 |
| Successfully processed | 3,696 |
| Rejected | 284 |

## Usage

```bash
python3 src/prepare_data.py
```

## Next Steps

- Implement CRF feature extraction and training
- Evaluate precision, recall, and F1-score
- Compare against a rule-based baseline
- Implement abbreviation-to-long-form linking

## Status

**In development** — preprocessing completed; model training and evaluation pending.
