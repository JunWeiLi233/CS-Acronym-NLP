import json
import re
from collections import Counter
from pathlib import Path

"""
    B-AC: Beginning of an abbreviation
    B-LF: Beginning of a long form
    I-LF: Inside a long form
    B-O: Other token(not an abbreviation or long form)
"""

LABELS = {"B-AC", "B-LF", "I-LF", "B-O"}

# keep punctuation as separate tokens in here
TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    # Return tokens with their character offsets
    return [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        }
        for match in TOKEN_PATTERN.finditer(text)
    ]



def get_spans(record, field):
    """Read annotations in the form: [[start,end],...]"""
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

    # check if it is complete tokens
    for token in tokens:
        overlaps = token["start"] < end and token["end"] > start
        contained = token["start"] >= start and token["end"] <= end
        if overlaps and not contained:
            raise ValueError("Partial token overlap detected")

    # labeling
    if kind == "AC":
        if len(matching) != 1:
            raise ValueError("Abbreviation spans must contain one token")
        new_labels = ["B-AC"]
    elif kind == "LF":
        new_labels = ["B-LF"] + ["I-LF"] * (len(matching) - 1)
    else:
        raise ValueError(f"Unknown annotation kind: {kind}")

    # wrap up labeling huzz---
    for i, label in zip(matching, new_labels):
        if labels[i] != "B-O":
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


def convert_record(record, index):
    """Convert one raw sample to token-level BIO labels"""
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object")

    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError("Missing or invalid text field")

    # Read annotations FIRST
    acronym_spans = get_spans(record, "acronyms")
    long_form_spans = get_spans(record, "long-forms")
    long_form_spans, corrections = repair_long_form_spans(text, long_form_spans)

    # Validate abbreviation spans
    previous_end = 0
    for start, end in sorted(acronym_spans):
        if end > len(text):
            raise ValueError("Acronym span exceeds text length")
        if start < previous_end:
            raise ValueError("Overlapping abbreviation spans")
        if not text[start:end].strip():
            raise ValueError("Empty abbreviation span")
        previous_end = end

    # Pass the abbreviation spans to tokenize()
    tokens = tokenize(text)
    labels = ["B-O"] * len(tokens)

    for start, end in acronym_spans:
        assign_span(tokens, labels, start, end, "AC")

    for start, end in long_form_spans:
        if end > len(text):
            raise ValueError("Long-form span exceeds text length")
        assign_span(tokens, labels, start, end, "LF")

    assert len(tokens) == len(labels)
    assert all(label in LABELS for label in labels)

    return {
        "id": str(record.get("id", index)),
        "tokens": [token["text"] for token in tokens],
        "labels": labels,
        "offsets": [[token["start"], token["end"]] for token in tokens],
        "text": text,
        "corrections": corrections
    }


def load_records(path):
    """Load a JSON array or a JSONL file"""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Expected a list of JSON records")
    return records



def prepare_file(input_path, output_path):
    """Prepare output/file"""
    records = load_records(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    errors = []

    for index, record in enumerate(records):
        try:
            converted.append(convert_record(record, index))
        except (ValueError, KeyError, TypeError) as error:
            errors.append({"index": index, "error": str(error)})

    error_path = output_path.with_suffix(".errors.json")
    partial_path = output_path.with_suffix(".partial.jsonl")

    # Choose the new output file
    new_output = partial_path if errors else output_path

    with new_output.open("w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Remove the old dataset
    old_output = output_path if errors else partial_path
    if old_output.exists():
        old_output.unlink()

    # Save error report
    with error_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print("\n===== Dataset Preparation Results =====")
    print(f"Total records: {len(records)}")
    print(f"Successfully converted: {len(converted)}")
    print(f"Rejected: {len(errors)}")
    print(f"Output: {new_output}")
    print(f"Error report: {error_path}")

    if errors:
        print("\nPartial dataset generated. Review rejected records before training.")

    return new_output


def inspect_failed_records(raw_path, error_path, limit=10):
    """Debugging Helper Function"""
    records = load_records(raw_path)
    with open(error_path, encoding="utf-8") as f:
        errors = json.load(f)

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

    print("\n=== Error Categories ===")
    for category, count in categories.items():
        print(f"{category}: {count}")

    print("\n=== Failed Record Details ===")
    for error in errors[:limit]:
        record = records[error["index"]]
        print(f"\nIndex: {error['index']}")
        print("Error:", error["error"])
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            print("Record:", record)
            continue

        text = record["text"]
        print("Text:", repr(text))
        for field in ("acronyms", "long-forms"):
            print(f"\n{field}:")
            try:
                spans = get_spans(record, field)
            except ValueError as error:
                print(error)
                continue

            for start, end in spans:
                print(f"  [{start}, {end}]: {text[start:end]!r}")
                print("  Context:", repr(text[max(0, start - 15):end + 15]))


def choose_dataset(raw_dir):
    files = sorted(
        file for file in raw_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in (".json", ".jsonl")
    )
    if not files:
        print(f"No datasets found in {raw_dir}")
        return None

    print("\nAvailable datasets:\n")
    for i, file in enumerate(files, start=1):
        print(f"[{i}] {file.relative_to(raw_dir)}")
    print("[0] Back")

    choice = input("\nSelect a dataset: ").strip()
    if choice == "0":
        return None
    try:
        index = int(choice) - 1
    except ValueError:
        print("Please enter a number.")
        return None
    if not 0 <= index < len(files):
        print("Invalid selection.")
        return None
    return files[index]


def main():
    # Find project root automatically
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"

    while True:
        print("\n===== NLP Dataset Tool =====")
        print("[1] Prepare dataset\n[2] Inspect failed records\n[0] Exit")
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
            output_path = (output_dir / relative_path).with_suffix(".jsonl")
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
