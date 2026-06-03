import json
import re
from collections import Counter
from  simplemma import simple_tokenizer, lemmatize

# Define input and output filenames
INPUT_FILE = "ingredient_frequencies.json"
OUTPUT_FILE = "word_frequencies.json"


def lemmatize_recipe_word(token):
    word = re.sub(r"[^a-z]", "", token.lower())
    if not word:
        return ""

    # Handle plural variations (-ies -> -y)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # Handle plural variations (-es matches for items like tomatoes, potatoes)
    if word.endswith("es") and word[:-2].endswith(("ch", "sh", "x", "s", "o")):
        return word[:-2]
    # General plural striping (-s) ensuring words like 'butter' or 'swiss' are left intact
    if word.endswith("s") and not word.endswith(("ss", "is", "us")):
        return word[:-1]

    return word


def transform_frequency_map():
    print(f"Reading from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            ingredient_frequencies = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE} in the current directory.")
        return

    word_counter = Counter()

    print(f"Processing {len(ingredient_frequencies)} unique ingredient entries...")
    for phrase, freq in ingredient_frequencies.items():
        tokens = re.split(r'[\s_\-\(\)\'\`\/\.\,\"]+', phrase.strip().lower())
        for token in tokens:
            if token:
                lemma = lemmatize(token, lang="en")                
                if lemma:
                    word_counter[lemma] += freq

    sorted_word_frequencies = dict(
        sorted(word_counter.items(), key=lambda item: item[1], reverse=True)
    )

    print(f"Writing aggregated words to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_word_frequencies, f, indent=4)

    print("Execution complete.")


if __name__ == "__main__":
    transform_frequency_map()
