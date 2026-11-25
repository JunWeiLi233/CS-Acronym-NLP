"""Find abbreviations in raw text and say what they stand for.

Usage:
    python src/predict.py                         # choose a saved CRF from the menu
    python src/predict.py --text "We train a convolutional neural network (CNN). The CNN ..."
    python src/predict.py --file paper.txt --model crf        # uses models/crf.pkl
    python src/predict.py --file paper.txt --model rules --show-labels

The whole input is treated as one document: a long form found anywhere in
it is attached to every use of that abbreviation (singular or plural).
Abbreviations with no definition in the text are listed as undefined.
"""

import argparse
import re
import sys
from pathlib import Path

from prepare_data import tokenize
from terminal_table import format_table
from utils import MODEL_DIR, OUTSIDE, SAVED_CRF_FILENAMES, link_pairs, strip_plural
from utils import format_labeled_tokens as render

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def to_records(text, doc_id="input"):
    """Split a passage into tokenized records in a single document."""
    records = []
    sentences = (sentence for sentence in SENT_SPLIT.split(text.strip()) if sentence.strip())
    for index, sentence in enumerate(sentences):
        tokens = [token["text"] for token in tokenize(sentence)]
        if tokens:
            records.append({
                "id": f"{doc_id}-{index}",
                "doc_id": doc_id,
                "tokens": tokens,
                "labels": [OUTSIDE] * len(tokens),
                "text": sentence,
            })
    return records


def run_model(records, model_name, model_path=None, with_detector=True):
    """Predict BIO labels and carry abbreviation tags across sentences."""
    if model_name == "rules":
        from rule_based import predict
        return predict(records, propagation="all", glossary=True, with_detector=with_detector)
    from crf_model import CRFModel
    model = CRFModel.load(model_path or MODEL_DIR / "crf.pkl")
    return model.predict(records, propagation="all")


def glossary(records, preds):
    """Return (abbreviation, long form, definition sentence, first use) rows."""
    definitions = {}
    first_uses = {}
    for sentence_number, (record, labels) in enumerate(zip(records, preds), start=1):
        tokens = record["tokens"]
        for abbreviation_span, (start, end) in link_pairs(tokens, labels):
            abbreviation = tokens[abbreviation_span[0]]
            long_form = " ".join(tokens[start:end])
            definitions.setdefault(strip_plural(abbreviation), (long_form, sentence_number))
        for token, label in zip(tokens, labels):
            if label == "B-AC":
                first_uses.setdefault(token, sentence_number)

    # Insertion order follows the first labeled mention of each spelling.
    rows = []
    for abbreviation, first_use in first_uses.items():
        long_form, definition_sentence = definitions.get(strip_plural(abbreviation), (None, None))
        rows.append((abbreviation, long_form, definition_sentence, first_use))
    return rows


def show_predictions(records, model_name, model_path, *, show_labels=False):
    predictions = run_model(records, model_name, model_path)

    if show_labels:
        print(format_table(
            ("Sentence", "Labeled tokens"),
            ((number, render(record["tokens"], labels))
             for number, (record, labels) in enumerate(zip(records, predictions), start=1)),
            title="Sentence labels", right=(0,), wrap=1,
        ))
        print()

    rows = glossary(records, predictions)
    if not rows:
        print("No abbreviations found.")
        return
    display_rows = []
    for abbreviation, long_form, definition_sentence, first_use in rows:
        defined_in = definition_sentence if definition_sentence is not None else "-"
        display_rows.append((abbreviation, long_form or "Undefined", first_use, defined_in))
    print(format_table(
        ("Abbreviation", "Long form", "First use", "Defined in"),
        display_rows,
        title="Abbreviations by sentence", right=(2, 3), wrap=1,
    ))


def run_menu():
    models = {
        "1": ("Plain CRF", SAVED_CRF_FILENAMES["plain"]),
        "2": ("CRF + Schwartz–Hearst", SAVED_CRF_FILENAMES["schwartz_hearst"]),
        "3": ("CRF + Kaggle lexicon", SAVED_CRF_FILENAMES["kaggle_lexicon"]),
    }
    rows = [(option, name) for option, (name, _) in models.items()]
    rows.append(("0", "Exit"))
    model_menu = format_table(("Option", "Model"), rows, title="Saved CRF models", right=(0,))
    input_menu = format_table(
        ("Option", "Input"), ((1, "Enter text"), (2, "Read a text file"), (0, "Back")), right=(0,),
    )

    while True:
        print("\n" + model_menu)
        choice = input("\nChoose a model: ").strip()
        if choice == "0":
            return
        if choice not in models:
            print("Invalid model selection.")
            continue

        name, filename = models[choice]
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            print(f"Saved model file not found: {model_path}")
            continue

        print(f"\n{name}\n{input_menu}")
        source = input("\nChoose an input: ").strip()
        if source == "0":
            continue
        try:
            if source == "1":
                text = input("\nText: ")
            elif source == "2":
                path = Path(input("\nText file path: ").strip()).expanduser()
                text = path.read_text(encoding="utf-8")
            else:
                print("Invalid input selection.")
                continue

            records = to_records(text)
            if not records:
                print("No text to label.")
                continue
            show_predictions(records, "crf", model_path)
        except (OSError, ValueError) as error:
            print(f"\nError: {error}")
        input("\nPress Enter to continue...")


def main():
    if not sys.argv[1:]:
        try:
            run_menu()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
        return

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to analyze")
    src.add_argument("--file", type=Path, help="UTF-8 text file to analyze")
    ap.add_argument("--model", choices=["rules", "crf"], default="crf", help="prediction model (default: crf)")
    ap.add_argument("--model-path", default=None, help="CRF model file (default: models/crf.pkl)")
    ap.add_argument("--show-labels", action="store_true", help="show token tags by sentence")
    args = ap.parse_args()

    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    records = to_records(text)
    if not records:
        sys.exit("No text to label.")
    show_predictions(records, args.model, args.model_path, show_labels=args.show_labels)


if __name__ == "__main__":
    main()
