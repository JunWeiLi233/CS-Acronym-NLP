"""Convert span annotations into token-level BIO records for model training."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from terminal_table import choose_dataset, format_table
from utils import DATA_DIR, LABELS, OUTSIDE, ROOT, load_jsonl, save_jsonl

# Words with internal hyphens or apostrophes stay together. Punctuation becomes
# its own token so annotation offsets remain precise.
TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    """Return tokens together with their character offsets in ``text``."""
    return [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in TOKEN_PATTERN.finditer(text)
    ]



def get_spans(record, field):
    """Return ``field`` annotations as validated ``(start, end)`` pairs."""
    spans = record.get(field, [])
    if not isinstance(spans, list):
        raise ValueError(f"{field} must be a list")

    result = []
    for span in spans:
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or not all(type(x) is int for x in span)
        ):
            raise ValueError(f"Invalid {field} span: {span}")

        start, end = span
        if start < 0 or end <= start:
            raise ValueError(f"Invalid span offsets: {span}")
        result.append((start, end))
    return result


def assign_span(tokens, labels, start, end, kind):
    """Assign BIO labels to the tokens covered by one span."""
    matching = [
        i for i, token in enumerate(tokens)
        if token["start"] >= start and token["end"] <= end
    ]
    if not matching:
        raise ValueError(f"No tokens aligned with span[{start}, {end}]")

    first = tokens[matching[0]]
    last = tokens[matching[-1]]
    if first["start"] != start or last["end"] != end:
        raise ValueError("Span does not align with token boundaries")

    # A character span must cover whole tokens. Otherwise BIO labels would no
    # longer identify the original annotated text.
    for token in tokens:
        overlaps = token["start"] < end and token["end"] > start
        contained = token["start"] >= start and token["end"] <= end
        if overlaps and not contained:
            raise ValueError("Partial token overlap detected")

    if kind == "AC":
        if len(matching) != 1:
            raise ValueError("Abbreviation spans must contain one token")
        new_labels = ["B-AC"]
    elif kind == "LF":
        new_labels = ["B-LF"] + ["I-LF"] * (len(matching) - 1)
    else:
        raise ValueError(f"Unknown annotation kind: {kind}")

    for i, label in zip(matching, new_labels):
        if labels[i] != OUTSIDE:
            raise ValueError(f"Overlapping annotations at token {i}")
        labels[i] = label


def repair_long_form_spans(text, spans):
    """Repair one-character truncations and keep a correction log."""
    repaired = []
    corrections = []

    for start, end in spans:
        new_end = end
        if 0 <= start < end < len(text):
            # Annotation ends inside a word
            inside_word = text[end - 1].isalnum() and text[end].isalnum()
            word_ends_next = end + 1 == len(text) or not text[end + 1].isalnum()
            if inside_word and word_ends_next:
                new_end = end + 1

        repaired.append((start, new_end))
        if new_end != end:
            corrections.append({
                "original": text[start:end],
                "corrected": text[start:new_end],
                "old_span": [start, end],
                "new_span": [start, new_end]
            })
    return repaired, corrections


def get_pairs(record, acronym_spans, long_form_spans):
    """Read explicit abbreviation/long-form links by their raw span indexes.

    ``pairs`` is optional so existing corpora continue to use the evaluation
    linker's inferred pairs. When supplied, even an empty list is meaningful:
    it says that none of the spans in this sentence form a definition pair.
    """
    if "pairs" not in record:
        return None
    pairs = record["pairs"]
    if not isinstance(pairs, list):
        raise ValueError("pairs must be a list")

    validated = []
    for pair in pairs:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(type(index) is int for index in pair)
        ):
            raise ValueError(f"Invalid pair indexes: {pair}")
        acronym_index, long_form_index = pair
        if not 0 <= acronym_index < len(acronym_spans):
            raise ValueError(f"Acronym pair index out of range: {acronym_index}")
        if not 0 <= long_form_index < len(long_form_spans):
            raise ValueError(f"Long-form pair index out of range: {long_form_index}")
        validated.append((acronym_index, long_form_index))
    return validated


def span_tokens(tokens, start, end):
    """Return the complete tokens aligned to a validated character span."""
    return [
        token["text"]
        for token in tokens
        if token["start"] >= start and token["end"] <= end
    ]


def convert_record(record, index):
    """Convert one annotated sentence into the project's JSONL record format."""
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object")

    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError("Missing or invalid text field")

    acronym_spans = get_spans(record, "acronyms")
    long_form_spans = get_spans(record, "long-forms")
    long_form_spans, corrections = repair_long_form_spans(text, long_form_spans)
    pairs = get_pairs(record, acronym_spans, long_form_spans)

    previous_end = 0
    for start, end in sorted(acronym_spans):
        if end > len(text):
            raise ValueError("Acronym span exceeds text length")
        if start < previous_end:
            raise ValueError("Overlapping abbreviation spans")
        if not text[start:end].strip():
            raise ValueError("Empty abbreviation span")
        previous_end = end

    tokens = tokenize(text)
    labels = [OUTSIDE] * len(tokens)

    for start, end in acronym_spans:
        assign_span(tokens, labels, start, end, "AC")

    for start, end in long_form_spans:
        if end > len(text):
            raise ValueError("Long-form span exceeds text length")
        assign_span(tokens, labels, start, end, "LF")

    if len(tokens) != len(labels) or not all(label in LABELS for label in labels):
        raise AssertionError("Token and BIO label output is inconsistent")

    converted = {
        "id": str(record.get("id", index)),
        "tokens": [token["text"] for token in tokens],
        "labels": labels,
        "offsets": [[token["start"], token["end"]] for token in tokens],
        "text": text,
        "corrections": corrections
    }
    if record.get("doc_id") is not None:
        converted["doc_id"] = record["doc_id"]
    if pairs is not None:
        converted["pairs"] = [
            [
                span_tokens(tokens, *acronym_spans[acronym_index])[0],
                " ".join(span_tokens(tokens, *long_form_spans[long_form_index])).lower(),
            ]
            for acronym_index, long_form_index in pairs
        ]
    return converted


def load_records(path):
    """Load a JSON array or JSONL file containing annotation records."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    with path.open(encoding="utf-8") as source:
        records = json.load(source)

    if not isinstance(records, list):
        raise ValueError("Expected a list of JSON records")
    return records



def prepare_file(input_path, output_path):
    """Convert an annotation file and write the accepted records and errors."""
    records = load_records(input_path)
    output_path = Path(output_path)
    converted = []
    errors = []

    for index, record in enumerate(records):
        try:
            converted.append(convert_record(record, index))
        except (ValueError, KeyError, TypeError) as error:
            errors.append({"index": index, "error": str(error)})

    error_path = output_path.with_suffix(".errors.json")
    partial_path = output_path.with_suffix(".partial.jsonl")

    new_output = partial_path if errors else output_path

    save_jsonl(converted, new_output)

    old_output = output_path if errors else partial_path
    if old_output.exists():
        old_output.unlink()

    with error_path.open("w", encoding="utf-8") as error_file:
        json.dump(errors, error_file, indent=2, ensure_ascii=False)

    print("\n" + format_table(
        ("Records", "Count"),
        (("Total", f"{len(records):,}"), ("Converted", f"{len(converted):,}"),
         ("Rejected", f"{len(errors):,}")),
        title="Dataset preparation results", right=(1,),
    ))
    print("\n" + format_table(
        ("File", "Path"), (("Dataset", new_output), ("Error report", error_path)), wrap=1,
    ))

    if errors:
        print("\nPartial dataset generated. Review rejected records before training.")

    return new_output


def inspect_failed_records(raw_path, error_path, limit=10):
    """Print a categorized sample of records rejected during conversion."""
    records = load_records(raw_path)
    with open(error_path, encoding="utf-8") as error_file:
        errors = json.load(error_file)

    if not errors:
        print("No failed records in the latest report.")
        return

    categories = Counter()
    for error in errors:
        message = error["error"]
        if "Span does not align" in message:
            categories["Boundary mismatch"] += 1
        elif "No tokens aligned" in message:
            categories["No aligned tokens"] += 1
        elif "Overlapping" in message:
            categories["Overlapping spans"] += 1
        else:
            categories["Other"] += 1

    print("\n" + format_table(
        ("Error category", "Records"), categories.most_common(), title="Rejected records", right=(1,),
    ))

    shown = errors[:limit]
    print(f"\nShowing {len(shown):,} of {len(errors):,} rejected records")
    for failure in shown:
        index = failure["index"]
        record = records[index]
        rows = [("Error", failure["error"])]
        if isinstance(record, dict) and isinstance(record.get("text"), str):
            text = record["text"]
            rows.append(("Text", repr(text)))
            for field in ("acronyms", "long-forms"):
                try:
                    spans = get_spans(record, field)
                except ValueError as error:
                    rows.append((field, str(error)))
                    continue
                for start, end in spans:
                    context = text[max(0, start - 15):end + 15]
                    rows.append((f"{field} [{start}, {end}]", f"{text[start:end]!r}\nContext: {context!r}"))
        else:
            rows.append(("Record", repr(record)))
        print("\n" + format_table(("Field", "Value"), rows, title=f"Failed record {index}", wrap=1))


def main():
    parser = argparse.ArgumentParser(description="Prepare span annotations as token-level BIO records.")
    parser.add_argument("--input", type=Path, help="source JSON or JSONL, for non-interactive conversion")
    parser.add_argument("--output", type=Path, help="destination JSONL for --input")
    args = parser.parse_args()
    if args.input or args.output:
        if not args.input or not args.output:
            parser.error("--input and --output must be supplied together")
        if args.output.suffix.lower() != ".jsonl":
            parser.error("--output must have a .jsonl extension")
        destinations = (args.output, args.output.with_suffix(".partial.jsonl"),
                        args.output.with_suffix(".errors.json"))
        if args.input.resolve() in {path.resolve() for path in destinations}:
            parser.error("the input must be separate from all output files")
        try:
            prepare_file(args.input, args.output)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return

    raw_dir = ROOT / "data" / "raw"

    while True:
        print("\n" + format_table(
            ("Option", "Action"), ((1, "Prepare dataset"), (2, "Inspect failed records"), (0, "Exit")),
            title="NLP Dataset Tool", right=(0,),
        ))
        action = input("\nChoose an action: ").strip()
        if action == "0":
            break
        if action not in ("1", "2"):
            print("Invalid action.")
            continue

        try:
            input_path = choose_dataset(raw_dir)
            if input_path is None:
                continue

            relative_path = input_path.relative_to(raw_dir)
            output_path = (DATA_DIR / relative_path).with_suffix(".jsonl")
            error_path = output_path.with_suffix(".errors.json")

            if action == "1":
                print(f"\nProcessing: {input_path.name}")
                prepare_file(input_path, output_path)
            elif error_path.is_file():
                inspect_failed_records(input_path, error_path, limit=20)
            else:
                print("No error report found. Run dataset preparation first.")
        except (ValueError, OSError) as error:
            print(f"\nError: {error}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
