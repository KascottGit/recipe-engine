import csv
import json
import os
import ast
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import spacy
from spacy.matcher import PhraseMatcher

from ingredient_parser import parse_single_ingredient
from tqdm import tqdm


COOKING_LEMMAS = {
    # Wet Heat (Water/Steam based - No Maillard reaction)
    "boil": "cooked",
    "simmer": "cooked",
    "poach": "cooked",
    "blanch": "cooked",
    "steam": "cooked",
    # Dry Heat (Oven/Air based - Maillard reaction)
    "bake": "cooked",
    "roast": "cooked",
    "toast": "cooked",
    "grill": "cooked",
    # Direct Heat + Fat (Maillard reaction + Lipid absorption)
    "fry": "cooked",
    "saute": "cooked",
    "sauté": "cooked",
    "sear": "cooked",
    "brown": "cooked",
    "caramelize": "cooked",
    # Generic / State Changes
    "cook": "cooked",
    "melt": "cooked",
    "smoke": "cooked",
}

BASE_INGREDIENTS = []

# Global variable for the worker processes
nlp = None
matcher = None
canonical_map = {}


def init_worker():
    global nlp, matcher, canonical_map

    # 1. Load the base NLP, disabling heavy statistical parser and tok2vec components
    # we only need lemmatizer/tagger (which require attribute_ruler) and sentence segmentation
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tok2vec"])
    nlp.add_pipe("sentencizer")
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
        p = parse_single_ingredient(ingredient)
        if p:
            parsed_ingredients.append(p)

    # 2. Extract Cooking States
    full_instructions = " ".join(directions_data)
    doc = nlp(full_instructions)

    # Pre-calculate cooking states and sentence words once
    global_states = []
    sent_states = []
    for sent in doc.sents:
        words = {token.text.lower() for token in sent}
        lemmas = [COOKING_LEMMAS[token.lemma_] for token in sent if token.lemma_ in COOKING_LEMMAS]
        if lemmas:
            global_states.extend(lemmas)
        sent_states.append((words, lemmas))

    terminal_state = global_states[-1] if global_states else "raw"

    # 3. Synthesize Compound Tokens
    recipe_tokens = []
    for base in parsed_ingredients:
        ingredient_state = None
        base_words = set(word for word in base.split("_") if len(word) > 2)

        # Scan sentences for proximity matching using fast set intersection
        for words, lemmas in sent_states:
            if base_words & words:
                if lemmas:
                    ingredient_state = lemmas[-1]

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

    optimal_workers = max(1, multiprocessing.cpu_count())
    print(f"Spinning up {optimal_workers} CPU workers...")

    total_processed = 0

    with open(output_filepath, "w", encoding="utf-8") as outfile, \
         tqdm(total=max_rows, desc="Parsing Recipes", unit="recipe") as pbar:
         
        with ProcessPoolExecutor(
            max_workers=optimal_workers, initializer=init_worker
        ) as executor:
            for batch in stream_csv_batches(input_filepath):
                if max_rows is not None:
                    remaining_rows = max_rows - total_processed
                    if remaining_rows <= 0:
                        break

                    if len(batch) > remaining_rows:
                        batch = batch[:remaining_rows]

                results = executor.map(process_single_row, batch)

                for result in results:
                    if result and result["tokens"]:
                        outfile.write(json.dumps(result) + "\n")
                    pbar.update(1)

                total_processed += len(batch)

                if max_rows is not None and total_processed >= max_rows:
                    break


if __name__ == "__main__":
    # Ensure this points to the CSV file you uploaded
    input_path = "RecipeNLG_dataset.csv"
    output_path = "parsed_recipes.jsonl"

    print(f"Starting CSV pipeline execution on {input_path}...")
    process_csv_corpus(input_path, output_path, max_rows=100000)
    print(f"Execution finished. Output saved to {output_path}")
