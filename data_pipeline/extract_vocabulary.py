import pandas as pd
import ast
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ingredient_parser import parse_single_ingredient

def slice_batch_worker(raw_ingredients_batch):
    """
    Worker bridge function. Maps an array chunk to your single-item
    parser within an isolated, GIL-free system process.
    """
    local_map = {}
    for item in raw_ingredients_batch:
        local_map[item] = parse_single_ingredient(item)
    return local_map


def extract_unique_ingredients_multicore(filepath, output_filepath, max_rows=None):
    print(f"--- Step 1: Extracting Raw String Frequencies (Limit: {max_rows}) ---")
    raw_counts = Counter()
    corrupted_rows = 0

    chunksize = 100000
    try:
        for chunk in pd.read_csv(
            filepath, usecols=["ingredients"], chunksize=chunksize, nrows=max_rows
        ):
            for row in chunk["ingredients"]:
                try:
                    ingredient_data = ast.literal_eval(row)
                    for item in ingredient_data:
                        raw_counts[item] += 1
                except (ValueError, SyntaxError, KeyError):
                    corrupted_rows += 1
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    unique_raw_strings = list(raw_counts.keys())
    total_strings = len(unique_raw_strings)

    if total_strings == 0:
        print("No ingredients found. Exiting.")
        return

    print(
        f"Found {total_strings} unique raw strings. Skipped {corrupted_rows} corrupted rows."
    )

    print("\n--- Step 2: Parallel Parsing via Standalone Module ---")
    # Balance batches dynamically across your 16 AMD threads
    items_per_chunk = max(1, total_strings // 200) if total_strings < 100000 else 10000
    chunks = [
        unique_raw_strings[i : i + items_per_chunk]
        for i in range(0, total_strings, items_per_chunk)
    ]

    translation_map = {}
    num_workers = 16

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(slice_batch_worker, chunk) for chunk in chunks]

        with tqdm(
            total=total_strings, desc="Processing Ingredients", unit="str"
        ) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    translation_map.update(result)
                    pbar.update(len(result))
                except Exception as e:
                    print(f"\nWorker chunk failed: {e}")

    print("\n--- Step 3: Reducing and Aggregating Final Frequencies ---")
    final_sliced_counts = Counter()
    for raw_string, count in raw_counts.items():
        clean_token = translation_map.get(raw_string, "")
        if clean_token:
            final_sliced_counts[clean_token] += count

    sorted_ingredients = dict(final_sliced_counts.most_common())
    print(f"Total Cleaned Unique Ingredients: {len(sorted_ingredients)}")

    with open(output_filepath, "w", encoding="utf-8") as outfile:
        json.dump(sorted_ingredients, outfile, indent=4)

    print(f"Successfully exported final frequencies to {output_filepath}")


if __name__ == "__main__":
    input_file = "RecipeNLG_dataset.csv"
    output_file = "ingredient_frequencies.json"
    ROW_LIMIT = 500000
    
    extract_unique_ingredients_multicore(input_file, output_file, max_rows=ROW_LIMIT)
