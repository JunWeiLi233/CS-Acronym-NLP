# Tester guide

## Run the bundled test

From the repository root, install the dependencies and the spaCy English model
once, then open the comparison menu.

```bash
python3 -m pip install -r requirements.txt spacy
python3 -m spacy download en_core_web_sm
python3 src/compare_systems.py
```

1. Choose **3. Test on your dataset with bundled SDU training data**.
2. Select `held-out-cs.jsonl` by its menu number.
3. Open the JSON file printed at the end for scores and predictions from all seven systems.

Option **3** loads the three trained CRFs included in `models/`.

To score your own labels with the saved CRFs, [prepare your JSON file](../../README.md#use-your-own-articles),
then select the resulting JSONL in option **3**.

## Spot-check the labels (optional)

Open the [raw annotations](../raw/held-out-cs.json) in a text editor. Pick a
sentence, read its `text`, and check that the marked abbreviations and long forms
match what the sentence says. Check that each pair connects the right two spans.
If something looks wrong, note the sentence `id` and what should change.
[JSON example](../../README.md#write-annotations-as-json).
