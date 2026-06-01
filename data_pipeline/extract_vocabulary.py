import csv
import ast
import json
from collections import Counter

def extract_unique_ingredients(filepath, output_filepath):
    print(f"Scanning {filepath} for unique NER entities...")
    
    ingredient_counts = Counter()
    corrupted_rows = 0
    total_processed = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_processed += 1
            if total_processed % 100000 == 0:
                print(f"Processed {total_processed} rows...")
                
            try:
                # Extract the stringified array
                ner_data = ast.literal_eval(row['NER'])
                
                for item in ner_data:
                    if isinstance(item, str) and item.strip():
                        # Standardize the text before counting to prevent duplicate variants
                        clean_item = item.lower().strip()
                        ingredient_counts[clean_item] += 1
                        
            except (ValueError, SyntaxError, KeyError):
                corrupted_rows += 1
                continue

    # Sort the dictionary by frequency (highest to lowest)
    sorted_ingredients = dict(ingredient_counts.most_common())
    
    print(f"\nExtraction complete.")
    print(f"Total Unique Ingredients found: {len(sorted_ingredients)}")
    print(f"Corrupted rows skipped: {corrupted_rows}")
    
    # Save the full frequency map to disk
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(sorted_ingredients, outfile, indent=4)
        
    print(f"Saved frequency map to {output_filepath}")

if __name__ == "__main__":
    # Ensure this points to your 2M dataset
    input_file = "RecipeNLG_dataset.csv" 
    output_file = "ingredient_frequencies.json"
    
    extract_unique_ingredients(input_file, output_file)