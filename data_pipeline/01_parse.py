import csv
import json
import os
import ast
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import spacy


COOKING_LEMMAS = {
    "cook": "cooked",
    "fry": "fried",
    "bake": "baked",
    "roast": "roasted",
    "boil": "boiled",
    "simmer": "simmered",
    "grill": "grilled",
    "brown": "browned",
    "sear": "seared",
    "toast": "toasted",
    "caramelize": "caramelized",
    "melt": "melted",
    "saute": "sautéed",
    "sauté": "sautéed",
    "smoke": "smoked",
    "blanch": "blanched",
    "poach": "poached",
    "steam": "steamed",
}

# Global variable for the worker processes
nlp = None


def init_worker():
    """
    Initializes the NLP model once per CPU core worker.
    We aggressively strip out 'tagger' and 'attribute_ruler' to speed up the pipeline.
    """
    global nlp
    nlp = spacy.load("en_core_web_sm", disable=["ner"])


def process_single_row(row):
    global nlp

    try:
        base_ingredients = ast.literal_eval(row["NER"])
        directions_data = ast.literal_eval(row["directions"])
    except (ValueError, SyntaxError, KeyError) as e:
        print(e)
        return None  # Skip corrupted rows

    # 1. Format the pre-cleaned ingredients for our vectors
    # Replaces spaces with underscores: "chicken breast" -> "chicken_breast"
    clean_bases = []
    for base in base_ingredients:
        if isinstance(base, str) and base.strip():
            clean_bases.append(base.lower().strip().replace(" ", "_"))

    # 2. Extract Cooking States
    full_instructions = " ".join(directions_data)
    doc = nlp(full_instructions)

    # Pre-calculate the terminal state of the dish
    global_states = [
        COOKING_LEMMAS[token.lemma_] for token in doc if token.lemma_ in COOKING_LEMMAS
    ]
    terminal_state = global_states[-1] if global_states else "raw"

    # 3. Synthesize Compound Tokens
    recipe_tokens = []
    for base in clean_bases:
        ingredient_state = None
        base_words = base.split("_")

        # Scan sentences for proximity matching
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(word in sent_text for word in base_words if len(word) > 2):
                for token in sent:
                    if token.lemma_ in COOKING_LEMMAS:
                        ingredient_state = COOKING_LEMMAS[token.lemma_]

        final_state = ingredient_state if ingredient_state else terminal_state
        recipe_tokens.append(f"{base}_{final_state}")

    return {
        "id": row.get("Row", row.get("", "unknown")),
        "title": row.get("Name", row.get("title", "Unknown Recipe")),
        "tokens": recipe_tokens,
    }


def stream_csv_batches(filepath, batch_size=5000):
    """Increased batch size to feed the multi-core executor more efficiently."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def process_csv_corpus(input_filepath, output_filepath, max_rows=None):
    os.makedirs(
        os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else ".",
        exist_ok=True,
    )

    # Leave 1 or 2 cores free so your OS doesn't freeze
    optimal_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Spinning up {optimal_workers} CPU workers...")

    total_processed = 0

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        # Initialize the process pool
        with ProcessPoolExecutor(
            max_workers=optimal_workers, initializer=init_worker
        ) as executor:
            for batch_num, batch in enumerate(stream_csv_batches(input_filepath)):
                # Enforce the max_rows limit
                if max_rows is not None:
                    remaining_rows = max_rows - total_processed
                    if remaining_rows <= 0:
                        break  # Safety catch

                    # If this batch pushes us over the limit, slice it down to the exact remainder
                    if len(batch) > remaining_rows:
                        batch = batch[:remaining_rows]

                print(f"Processing Batch {batch_num + 1} ({len(batch)} rows)...")

                # Execute the batch in parallel
                results = executor.map(process_single_row, batch)

                # Write results sequentially as they finish
                for result in results:
                    if result and result["tokens"]:  # Ignore skipped/empty rows
                        outfile.write(json.dumps(result) + "\n")

                total_processed += len(batch)

                # Halt the generator if the limit is hit
                if max_rows is not None and total_processed >= max_rows:
                    print(f"\nReached max_rows limit ({max_rows}). Halting extraction.")
                    break


if __name__ == "__main__":
    # Ensure this points to the CSV file you uploaded
    input_path = "RecipeNLG_dataset.csv"
    output_path = "parsed_recipes.jsonl"

    print(f"Starting CSV pipeline execution on {input_path}...")
    process_csv_corpus(input_path, output_path, max_rows=10000)
    print(f"Execution finished. Output saved to {output_path}")
