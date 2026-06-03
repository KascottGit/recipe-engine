import re
import simplemma
from ingredient_slicer import IngredientSlicer

STOPWORDS = {
    "skinless",
    "boneless",
    "fresh",
    "organic",
    "frozen",
    "original",
    "each",
    "cut",
    "pieces",
    "bonelessskinless",
    "skinlessboneless",
    "halve",
    "half",
    "bonedskinless",
    "bone",
    "mild",
    "warm",
    "lukewarm",
    "uncle",
    "ben",
    "raw",
    "parcooked",
    "cooked",
    "prepare",
    "skin",
    "split",
    "halfway",
    "way",
    "longway",
    "longways",
    "heinz",
    "good",
    "size",
    "bite",
    "tear",
    "wash",
    "hand",
    "handtorn",
    "hull",
    "hulless",
    "desire",
    "dairy",
    "nondairy",
    "plain",
    "low",
    "lowfat",
    "halfdollar",
    "third",
    "break",
    "halvesthirds",
    "crisp",
    "crispy",
    "dollar",
    "moist",
    "moisten",
    "extra",
    "process",
    "nonfat",
    "dry",
    "commercial",
    "soft",
    "hard",
    "doz",
    "soften",
    "cook",
    "cooked",
    "noncooked",
    "quick",
    "uncooked",
    "uncook",
    "minute",
    "regular",
    "roll",
    "quaker",
    "quickcooking",
    "grain",
    "substitute",
    "artificial",
    "equal",
    "sweeten",
    "unsweeten",
    "brand",
    "eagle",
    "sweetener",
    "tub",
    "lite",
    "slight",
    "semisweet",
    "baker",
    "german",
    "direction",
    "seedless",
    "dissolve",
    "bit",
    "tidbit",
    "duncan",
    "hines",
    "devil",
    "underwood",
    "chunky",
    "frenchstyle",
    "mixe",
    "knorr",
    "reserve",
    "drained",
    "undrained",
    "undiluted",
    "heat",
    "skor",
    "separate",
    "durkee",
    "pound",
    "thick",
    "thickness",
    "depend",
    "taste",
    "suit",
    "stuff",
    "stove",
    "purchase",
    "homemade",
    "homestyle",
    "hormel",
    "real",
    "fine",
    "snip",
    "wee",
    "semi",
    "morsel",
    "square",
    "double",
    "acting",
    "doubleacting",
    "type",
    "leftover",
    "instant",
    "pulp",
    "parkay",
    "pasteurize",
    "kraft",
    "pod",
    "ready",
    "keebler",
    "peeled",
    "unpeeled",
    "fryer"
}

BASE_INGREDIENTS_PRIORITY = [
    ("oil", "oil"),
    ("cereal", "cereal"),
    ("chicken", "chicken"),
    ("bacon", "bacon"),
    ("bakon", "bacon"),
    ("beef", "beef"),
    ("pork", "pork"),
    ("olive", "olive"),
    ("vegetable", "vegetable"),
    ("veg", "vegetable"),
    ("ketchup", "ketchup"),
    ("tomato", "tomato"),
    ("onion", "onion"),
    ("garlic", "garlic"),
    ("salt", "salt"),
    ("sugar", "sugar"),
    ("cheese", "cheese"),
    ("cheddar", "cheese"),  # Force 'cheddar' to map to 'cheese' base
    ("worcestershire", "worcestershire"),
    ("accent", "msg"),
    ("msg", "msg"),
    ("monosodium", "msg"),
    ("glutamate", "msg"),
    ("oatmeal", "oat"),
    ("oat", "oat"),
    ("grape", "grape"),
    ("corn", "corn"),
]

VARIANT_MAP = {
    "breast": "breast",
    "thigh": "thigh",
    "wing": "wing",
    "mince": "mince",
    "ground": "mince",
    "boullion": "broth",
    "bouillon": "broth",
    "broth": "broth",
    "stock": "broth",
    "paste": "paste",
    "powder": "powder",
    "olive": "olive",
    "chicken": "chicken",
    "beef": "beef",
    "flavoring": "flavor",
    "catsup": "ketchup",
}

PHRASE_REPLACEMENTS = {
    "monosodium glutamate": "msg",
    "lea perrins": "worcestershire",
    "lea and perrins": "worcestershire",
    "sour cream": "sour_cream",  # Forces compound nouns to stay together
    "brown sugar": "brown_sugar",
    "tomato ketchup": "ketchup",
    "tomato catsup": "ketchup",
    "baking soda": "baking_soda",
    "heath": "health",
}

sorted_phrases = sorted(PHRASE_REPLACEMENTS.keys(), key=len, reverse=True)
PHRASE_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, sorted_phrases)) + r")\b", re.IGNORECASE
)


def parse_single_ingredient(item: str) -> str:

    if not isinstance(item, str) or not item.strip():
        return ""

    try:
        slicer = IngredientSlicer(item)
        food_attr = slicer.food
        ingredient = food_attr() if callable(food_attr) else food_attr

        if not ingredient:
            return ""

        # 1. Multi-Word Phrase Replacement
        cleaned_text = ingredient.lower().strip()
        cleaned_text = PHRASE_PATTERN.sub(
            lambda mo: PHRASE_REPLACEMENTS[mo.group(0).lower()], cleaned_text
        )

        # 2. Tokenize and Lemmatize
        raw_words = cleaned_text.split()
        all_lemmas = []
        for w in raw_words:
            cleaned_word = "".join(e for e in w if e.isalnum() or e == "_")
            if cleaned_word:
                all_lemmas.append(simplemma.lemmatize(cleaned_word, lang="en"))

        # 3. Filter Stopwords After Lemmatization
        lemmas = [lemma for lemma in all_lemmas if lemma not in STOPWORDS]

        if not lemmas:
            return ""

        found_base = None
        found_variant = None

        # 4. Base Anchor Extraction
        for target_word, token_name in BASE_INGREDIENTS_PRIORITY:
            if target_word in lemmas:
                found_base = token_name
                lemmas = [l for l in lemmas if l != target_word]
                break

        # 5. Variant Extraction
        for l in lemmas:
            if l in VARIANT_MAP:
                found_variant = VARIANT_MAP[l]
                lemmas = [rem for rem in lemmas if rem != l]
                break

        # 6. Structural Fallback Reconstruction
        rest_of_string = "_".join(lemmas)

        if found_base and found_variant:
            if found_base == found_variant:
                token = (
                    found_base
                    if not rest_of_string
                    else f"{found_base}_{rest_of_string}"
                )
            else:
                token = f"{found_base}_{found_variant}"
        elif found_base:
            token = f"{found_base}_{rest_of_string}" if rest_of_string else found_base
        elif found_variant:
            token = (
                f"{rest_of_string}_{found_variant}" if rest_of_string else found_variant
            )
        else:
            token = rest_of_string

        # 7. Order-Preserving Deduplication Pass
        final_sub_tokens = token.strip("_").split("_")
        deduplicated_tokens = list(dict.fromkeys(final_sub_tokens))

        return "_".join(deduplicated_tokens)

    except Exception:
        return ""
