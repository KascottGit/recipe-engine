import os
import psycopg2
from psycopg2.extras import execute_values

DB_DSN = os.environ.get(
    "DATABASE_URL", 
    "host=localhost dbname=recipedb user=admin password=supersecretpassword"
)


def classify_token(token):
    # Default is vegan, vegetarian, no gluten, no dairy, no nuts
    is_vegan = True
    is_vegetarian = True
    contains_gluten = False
    contains_dairy = False
    contains_nuts = False
    
    t_lower = token.lower()
    words = set(t_lower.split("_"))
    
    # Meat & Poultry
    meat_words = {
        "chicken", "beef", "pork", "bacon", "turkey", "ham", "steak", "veal", 
        "sausage", "pepperoni", "meat", "lamb", "mutton", "duck", "goose", 
        "venison", "bologna", "salami"
    }
    # Seafood
    seafood_words = {
        "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "prawn", 
        "clam", "mussel", "anchovy", "oyster", "scallop", "cod", "halibut", 
        "sardine", "haddock", "calamari", "squid"
    }
    # Dairy
    dairy_words = {
        "milk", "cheese", "butter", "cream", "yogurt", "whey", "ghee", 
        "buttermilk"
    }
    # Egg
    egg_words = {"egg", "yolk", "eggyolk", "eggwhite"}
    # Honey
    honey_words = {"honey"}
    # Gluten
    gluten_words = {
        "wheat", "flour", "barley", "rye", "semolina", "spelt", "macaroni", 
        "pasta", "spaghetti", "noodle"
    }
    # Nuts
    nut_words = {
        "peanut", "almond", "walnut", "cashew", "pecan", "pistachio", 
        "hazelnut", "macadamia", "nut"
    }

    if words & meat_words or words & seafood_words:
        is_vegan = False
        is_vegetarian = False
    elif words & egg_words:
        is_vegan = False
        is_vegetarian = True
    elif words & dairy_words:
        is_vegan = False
        is_vegetarian = True
        contains_dairy = True
    elif words & honey_words:
        is_vegan = False
        is_vegetarian = True

    if words & gluten_words:
        contains_gluten = True
        
    if any(nw in t_lower for nw in nut_words):
        contains_nuts = True
        
    return (token, is_vegan, is_vegetarian, contains_gluten, contains_dairy, contains_nuts)


def seed_taxonomy():
    print("1. Connecting to PostgreSQL database...")
    try:
        conn = psycopg2.connect(DB_DSN)
        cursor = conn.cursor()

        print("2. Ensuring ingredient_taxonomy table exists...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ingredient_taxonomy (
            token VARCHAR(255) PRIMARY KEY,
            is_vegan BOOLEAN DEFAULT TRUE,
            is_vegetarian BOOLEAN DEFAULT TRUE,
            contains_gluten BOOLEAN DEFAULT FALSE,
            contains_dairy BOOLEAN DEFAULT FALSE,
            contains_nuts BOOLEAN DEFAULT FALSE
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        print("3. Fetching unique tokens from ingredient_vectors...")
        cursor.execute("SELECT token FROM ingredient_vectors;")
        rows = cursor.fetchall()
        tokens = [row[0] for row in rows]
        
        print(f"   Found {len(tokens)} unique ingredient tokens in database.")
        if not tokens:
            print("   No tokens found. Please run 02_train_w2v.py first to load vectors.")
            return

        print("4. Categorizing tokens based on taxonomy rules...")
        classified_data = [classify_token(token) for token in tokens]

        print("5. Upserting taxonomy data into PostgreSQL...")
        insert_query = """
            INSERT INTO ingredient_taxonomy (
                token, is_vegan, is_vegetarian, contains_gluten, contains_dairy, contains_nuts
            ) 
            VALUES %s 
            ON CONFLICT (token) DO UPDATE SET 
                is_vegan = EXCLUDED.is_vegan,
                is_vegetarian = EXCLUDED.is_vegetarian,
                contains_gluten = EXCLUDED.contains_gluten,
                contains_dairy = EXCLUDED.contains_dairy,
                contains_nuts = EXCLUDED.contains_nuts;
        """
        
        execute_values(cursor, insert_query, classified_data, page_size=5000)
        conn.commit()
        print("Success! Ingredient taxonomy is successfully seeded in the database.")

    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    seed_taxonomy()
