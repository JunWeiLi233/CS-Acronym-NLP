# Abbreviation Detection Using CRF

Extracts abbreviations and their long forms from scientific text, such as
`NLP` and `Natural language processing`. This is a final project for NYU
Natural Language Processing, Fall 2026.

## Quick start

From the repository root, install the dependencies and open the comparison menu.

```bash
python3 -m pip install -r requirements.txt
python3 src/compare_systems.py
```

| Menu option | What it does |
| --- | --- |
| **1. Bundled SDU development split** | Trains the three CRFs on 2,956 sentences and scores all seven systems on 740 development sentences. |
| **2. Split your dataset** | Splits your processed data into training and development sets. Sentences with the same `doc_id` stay together. |
| **3. Test with bundled SDU training data** | Uses the three saved CRFs to score all seven systems on the bundled test or your own labeled JSONL. |
| **4. Choose training and test datasets** | Fits on the selected training file and scores on the separate test file. |

Option 3 reuses the compatible saved CRFs in `models/`. The comparison trains
fresh models if those files are unavailable or you request different training
settings. Option 3 saves scores and predictions to `results/<dataset>-comparison.json`
and prints the path. Repeat runs get a numbered filename.

To extract abbreviations from new text, run the rule-based predictor.

```bash
python3 src/predict.py --model rules --text "Natural language processing (NLP) is useful."
```

Use `--file document.txt` to read a text file or `--show-labels` to see token labels.

## Use your own articles

1. Save one JSON file under `data/raw/` using the [format below](#write-annotations-as-json).
2. Run `python3 src/prepare_data.py`. Choose **Prepare dataset** and select the file.
   Check the error report if the tool produces a `.partial.jsonl` file.
3. Run `python3 src/compare_systems.py` and select the resulting JSONL under
   `data/processed/` using option **2**, **3**, or **4**.

### Write annotations as JSON

Use one record per sentence and a unique `id` for each record. Give all
sentences from one article the same `doc_id`.

```json
[
  {
    "id": "paper1-s01",
    "doc_id": "paper1",
    "text": "Natural language processing (NLP) is useful.",
    "acronyms": [[29, 32]],
    "long-forms": [[0, 27]],
    "pairs": [[0, 0]]
  }
]
```

Spans use zero-based `[start, end]` character offsets, with the end excluded.
Each pair is `[acronym span index, long-form span index]`. Use `[]` for an empty
field. For older files without a `pairs` field, the evaluator infers links from
the spans.

For independent review of the held-out set, give each reviewer a copy of the
[blank labels](data/annotation/held-out-cs-blank.json). The [tester guide](data/annotation/review.md)
covers running the test and checking the labels.

## Train and use a CRF

Open the saved-model menu with `python3 src/predict.py`. Choose the plain CRF,
Schwartz–Hearst CRF, or Kaggle-lexicon CRF, then enter text or a text-file path.

To train a new model and use it from the command line, run these commands.

```bash
python3 src/train_crf.py
python3 src/predict.py --model crf --file document.txt
```

Training uses the bundled SDU split and saves the model to `models/crf.pkl`.
Use `--data` for a different processed JSONL, `--with-rules` for Schwartz–Hearst
features, or `--with-csit` for the CS & IT lexicon. Use `--model-out` to save a
variant separately and `--model-path` to load it for prediction.

For the part-of-speech features used in the recorded runs, install spaCy and its
English model.

```bash
python3 -m pip install spacy
python3 -m spacy download en_core_web_sm
```

## Systems

| System | How it works |
| --- | --- |
| Majority baseline | Marks every token as outside an abbreviation or long form. |
| Heuristic baseline | Guesses abbreviations from capitalization, punctuation, and an English word list. |
| Regex | Finds common definition patterns and marks later mentions. |
| Schwartz–Hearst + propagation | Matches abbreviation letters to an expansion and marks later mentions in the same document. |
| CRF | Learns token labels from spelling, context, and definition cues. |
| CRF + Schwartz–Hearst features | Gives the CRF extra clues from rule matches. |
| CRF + Kaggle lexicon | Adds a CS & IT acronym lookup to the plain CRF. |

The labels are `B-AC` for abbreviations, `B-LF` and `I-LF` for the start and
continuation of a long form, and `B-O` for other tokens. The comparison
reports precision, recall, and F1 for exact abbreviation spans, long-form
spans, and definition pairs.

Run `python3 src/compare_systems.py --with-csit` to include an eighth system
that combines Schwartz–Hearst and lexicon features.

## Datasets

| Dataset | Usable sentences | Use |
| --- | ---: | --- |
| [Main SDU](data/processed/SDU-train-data.partial.jsonl) | 3,696 | Default training and development data |
| [Supplemental SDU](data/processed/SDU22-scientific-extra-training.partial.jsonl) | 455 | Extra training examples |
| [Synthetic CS & IT](data/processed/csit-synthetic-training.jsonl) | 348 | Template-based training examples |
| [Held-out CS abstracts](data/processed/held-out-cs.jsonl) | 172 | Test set from 24 published abstracts, with preliminary Codex labels |
| SciAI | 1,639 | External human-annotated span benchmark |
| Ab3P | 11,062 | External human-annotated pair benchmark from 1,250 documents |

The main and supplemental SDU files have preprocessing reports for their
[284](data/processed/SDU-train-data.errors.json) and
[42](data/processed/SDU22-scientific-extra-training.errors.json) rejected records,
respectively.

## Dev results

Run the tuned comparison with CRFsuite and save the scores as JSON.

```bash
python3 src/compare_systems.py --backend crfsuite --c1 0.2 --c2 0.1 --output results/dev-comparison.json
```

After verifying a run, update the score tables and add a timeline row by hand.
Add `--extra-train data/processed/SDU22-scientific-extra-training.partial.jsonl`
to test supplemental training on the same development split.

The tables show the original six-system run from **2026-11-24 05:15:57 UTC**
using 2,956 training and 740 development sentences. S&H means Schwartz–Hearst.

### F1 scores

| System | Abbreviation | Long form | Pair |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.000 | 0.000 | 0.000 |
| Heuristic baseline | 0.908 | 0.000 | 0.000 |
| Regex | 0.879 | 0.682 | 0.658 |
| S&H + propagation | 0.759 | 0.772 | 0.724 |
| CRF | 0.943 | 0.834 | 0.813 |
| CRF + S&H features | 0.943 | 0.845 | 0.823 |

<details>
<summary>Precision and recall</summary>

### Precision

| System | Abbreviation | Long form | Pair |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.000 | 0.000 | 0.000 |
| Heuristic baseline | 0.848 | 0.000 | 0.000 |
| Regex | 0.867 | 0.955 | 0.911 |
| S&H + propagation | 0.977 | 0.870 | 0.807 |
| CRF | 0.930 | 0.842 | 0.826 |
| CRF + S&H features | 0.935 | 0.852 | 0.835 |

### Recall

| System | Abbreviation | Long form | Pair |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.000 | 0.000 | 0.000 |
| Heuristic baseline | 0.976 | 0.000 | 0.000 |
| Regex | 0.892 | 0.530 | 0.515 |
| S&H + propagation | 0.621 | 0.694 | 0.656 |
| CRF | 0.955 | 0.827 | 0.800 |
| CRF + S&H features | 0.952 | 0.837 | 0.812 |

</details>

### Dev results timeline

| Recorded (UTC) | Leading system | Pair F1 | Training data | CRF settings |
| --- | --- | ---: | --- | --- |
| 2026-11-24 05:15:40 | CRF + S&H features | 0.817079 | [SDU train split](data/processed/SDU-train-data.partial.jsonl) (2,956) | `--backend crfsuite --c1 0.05 --c2 0.05 --max-iter 150` (POS on) |
| 2026-11-24 05:15:57 | CRF + S&H features | 0.823416 | [SDU train split](data/processed/SDU-train-data.partial.jsonl) (2,956) | `--backend crfsuite --c1 0.2 --c2 0.1 --max-iter 150` (POS on) |

Adding 455 supplemental SDU sentences changed pair F1 from 0.813 to 0.807 for
the plain CRF and from 0.823 to 0.821 for the hybrid.

## Held-out and external results

In the original held-out run, the hybrid reached 0.950 abbreviation F1.
Regex and Schwartz–Hearst tied at 0.873 pair F1. On SciAI, the hybrid reached
0.932 abbreviation F1 and 0.893 long-form F1. On Ab3P, Schwartz–Hearst reached
0.826 pair F1.

<details>
<summary>Re-run the recorded test settings</summary>

Install the spaCy model, then run the held-out comparison.

```bash
python3 src/compare_systems.py --data data/processed/SDU-train-data.partial.jsonl \
  --test data/processed/held-out-cs.jsonl --backend crfsuite --c1 0.2 --c2 0.1 \
  --crf-groups form,ortho,context,pos,dict,defmatch \
  --hybrid-groups ortho,context,paren,pos,dict,defmatch,rules
```

For SciAI, use `--test results/sciai/sciai21-truth.jsonl --spans-only`. For
Ab3P, use `--test results/external-benchmark/ab3p-bioc.jsonl --pairs-only`.
Both external runs need the prepared JSONL files, which are kept locally.

</details>

## Authors

Junwei Li, Terence Li, Oscar Lin, and Sam Sutton.

## Sources

- **SDU and MACRONYM.** Veyseh et al. (2022). [Shared-task paper](https://ceur-ws.org/Vol-3164/paper31.pdf) ·
  [Corpus paper](https://aclanthology.org/2022.coling-1.292/) ·
  [Training file](https://github.com/amirveyseh/AAAI-22-SDU-shared-task-1-AE/blob/4a8f3b644c2824cf62e06907c8324fedc0941f24/data/english/scientific/train.json) ·
  [Development file](https://github.com/amirveyseh/AAAI-22-SDU-shared-task-1-AE/blob/4a8f3b644c2824cf62e06907c8324fedc0941f24/data/english/scientific/dev.json) ·
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Held-out abstracts.** ACL, EMNLP, and NAACL Findings 2025. [Source list](data/annotation/held-out-cs-sources.json) ·
  [Raw annotations](data/raw/held-out-cs.json).
- **External benchmarks.** [SciAI (Veyseh et al., 2020)](https://aclanthology.org/2020.coling-main.292/) ·
  [Ab3P (Sohn et al., 2008)](https://doi.org/10.1186/1471-2105-9-402).
- **CS & IT acronyms.** Youxise (`youcmoulai`), 2024. [Bundled lexicon](data/lexicon/csit_acronyms.json) ·
  [Kaggle](https://www.kaggle.com/datasets/youcmoulai/cs-and-it-acronyms) ·
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **English word list.** Kevin Atkinson’s SCOWL. [Project copy](data/lexicon/english_words.txt) ·
  [Upstream dictionaries](https://wordlist.aspell.net/dicts/) · [License](data/lexicon/LICENSE-SCOWL.txt).
- **Project seed lexicon.** [Bundled JSON](data/lexicon/cs_abbreviations_training_lexicon.json) ·
  [Source papers](#source-papers-for-the-seed-lexicon).
- **Sequence labeling and CRFs.** [Lafferty et al. (2001)](https://www.cis.upenn.edu/~danroth/Teaching/CS598-05/Papers/crf.pdf) ·
  [Sha and Pereira (2003)](https://aclanthology.org/N03-1028/) ·
  [CoNLL-2000 chunking](https://aclanthology.org/W00-0726/) ·
  [Abbreviation BIO labels](https://aclanthology.org/2024.lrec-main.270/) ·
  [CRFsuite](https://www.chokkan.org/software/crfsuite/) · [Python bindings](https://github.com/scrapinghub/python-crfsuite).
- **Abbreviation extraction.** [Schwartz and Hearst (2003)](https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf) ·
  [Kreuzthaler et al. (2016)](https://aclanthology.org/W16-4213/) ·
  [Vanopstal et al. (2010)](https://aclanthology.org/L10-1510/).
- **Part-of-speech tagging.** [spaCy](https://github.com/explosion/spaCy/blob/master/CITATION.cff) ·
  [Model card](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0) ·
  [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19).

### Source papers for the seed lexicon

- [BERT](https://aclanthology.org/N19-1423/)
- [Dynamo](https://www.amazon.science/publications/dynamo-amazons-highly-available-key-value-store)
- [Distributed machine learning](https://www.engineering.org.cn/engi/EN/1159829003290862291)
- [The Google File System](https://research.google/pubs/the-google-file-system/)
