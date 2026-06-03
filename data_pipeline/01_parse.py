import csv
import json
import os
import ast
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import spacy
from spacy.matcher import PhraseMatcher

from ingredient_parser import parse_single_ingredient


COOKING_LEMMAS = {
    # Wet Heat (Water/Steam based - No Maillard reaction)
    "boil": "boiled",
    "simmer": "boiled",
    "poach": "boiled",
    "blanch": "boiled",
    "steam": "boiled",
    # Dry Heat (Oven/Air based - Maillard reaction)
    "bake": "baked",
    "roast": "baked",
    "toast": "baked",
    "grill": "baked",
    # Direct Heat + Fat (Maillard reaction + Lipid absorption)
    "fry": "fried",
    "saute": "fried",
    "sauté": "fried",
    "sear": "fried",
    "brown": "fried",
    "caramelize": "fried",
    # Generic / State Changes
    "cook": "cooked",
    "melt": "melted",
    "smoke": "smoked",
}

BASE_INGREDIENTS = []

# Global variable for the worker processes
nlp = None
matcher = None
canonical_map = {}


def init_worker():
    global nlp, matcher, canonical_map

    # 1. Load the base NLP, stripping the useless default NER
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # Match on lowercase


def process_single_row(row):
    global nlp, matcher, canonical_map

    try:
        ingredients_data = ast.literal_eval(row["NER"])
        directions_data = ast.literal_eval(row["directions"])
    except (ValueError, SyntaxError, KeyError) as e:
        print(e)
        return None  # Skip corrupted rows

    parsed_ingredients = []
    for ingredient in ingredients_data:
        parsed_ingredients.append(parse_single_ingredient(ingredient))

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
    for base in parsed_ingredients:
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
    """
    recipe_tokens = []
    for base in parsed_ingredients:
        recipe_tokens.append(base)
    """

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

    optimal_workers = max(1, multiprocessing.cpu_count())
    print(f"Spinning up {optimal_workers} CPU workers...")

    total_processed = 0

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        with ProcessPoolExecutor(
            max_workers=optimal_workers, initializer=init_worker
        ) as executor:
            for batch_num, batch in enumerate(stream_csv_batches(input_filepath)):
                if max_rows is not None:
                    remaining_rows = max_rows - total_processed
                    if remaining_rows <= 0:
                        break

                    if len(batch) > remaining_rows:
                        batch = batch[:remaining_rows]

                print(f"Processing Batch {batch_num + 1} ({len(batch)} rows)...")

                results = executor.map(process_single_row, batch)

                for result in results:
                    if result and result["tokens"]:
                        outfile.write(json.dumps(result) + "\n")

                total_processed += len(batch)

                if max_rows is not None and total_processed >= max_rows:
                    print(f"\nReached max_rows limit ({max_rows}). Halting extraction.")
                    break


if __name__ == "__main__":
    # Ensure this points to the CSV file you uploaded
    input_path = "RecipeNLG_dataset.csv"
    output_path = "parsed_recipes.jsonl"

    print(f"Starting CSV pipeline execution on {input_path}...")
    process_csv_corpus(input_path, output_path, max_rows=100000)
    print(f"Execution finished. Output saved to {output_path}")
