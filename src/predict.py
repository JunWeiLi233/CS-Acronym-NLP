"""Find abbreviations in raw text and say what they stand for.

Usage:
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

from prepare_data import tokenize
from utils import MODEL_DIR, OUTSIDE, link_pairs, strip_plural

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def to_records(text, doc_id="input"):
    records = []
    for k, sent in enumerate(s for s in SENT_SPLIT.split(text.strip()) if s.strip()):
        tokens = [t["text"] for t in tokenize(sent)]
        if tokens:
            records.append({"id": f"{doc_id}-{k}", "doc_id": doc_id, "tokens": tokens,
                            "labels": [OUTSIDE] * len(tokens), "text": sent})
    return records


def run_model(records, model_name, model_path=None, with_detector=True):
    if model_name == "rules":
        from rule_based import predict
        return predict(records, propagation="all", glossary=True, with_detector=with_detector)
    from crf_model import CRFModel
    model = CRFModel.load(model_path or MODEL_DIR / "crf.pkl")
    return model.predict(records, propagation="all")


def glossary(records, preds):
    """abbreviation -> (long form, sentence number) plus undefined uses."""
    defs, uses = {}, {}
    for n, (r, p) in enumerate(zip(records, preds), start=1):
        for a, lf in link_pairs(r["tokens"], p):
            sf = r["tokens"][a[0]]
            defs.setdefault(strip_plural(sf), (" ".join(r["tokens"][lf[0]:lf[1]]), n))
        for t, l in zip(r["tokens"], p):
            if l == "B-AC":
                uses.setdefault(t, n)
    rows = []
    for sf, first_use in sorted(uses.items(), key=lambda x: x[1]):
        d = defs.get(strip_plural(sf))
        rows.append((sf, d[0] if d else None, d[1] if d else None, first_use))
    return rows


def render(tokens, labels):
    return " ".join(t if l == OUTSIDE else f"[{t}|{l[2:]}]" for t, l in zip(tokens, labels))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file")
    ap.add_argument("--model", choices=["rules", "crf"], default="crf")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--show-labels", action="store_true")
    args = ap.parse_args()

    text = args.text if args.text is not None else open(args.file, encoding="utf-8").read()
    records = to_records(text)
    if not records:
        sys.exit("No text to label.")
    preds = run_model(records, args.model, args.model_path)

    if args.show_labels:
        for i, (r, p) in enumerate(zip(records, preds), start=1):
            print(f"{i:3d}: {render(r['tokens'], p)}")
        print()

    rows = glossary(records, preds)
    if not rows:
        print("No abbreviations found.")
        return
    width = max(len(r[0]) for r in rows) + 2
    print(f"{'Abbreviation':{max(width, 14)}}Long form")
    for sf, lf, def_sent, first_use in rows:
        meaning = f"{lf}  (defined in sentence {def_sent})" if lf else "-- not defined in this text"
        print(f"{sf:{max(width, 14)}}{meaning}")


if __name__ == "__main__":
    main()
