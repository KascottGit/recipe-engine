import json
import multiprocessing
import psycopg2
from psycopg2.extras import execute_values
from gensim.models import Word2Vec

# Database connection string (from your earlier local Docker setup)
DB_DSN = "host=localhost dbname=recipedb user=admin password=supersecretpassword"


class StreamRecipeCorpus:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    recipe = json.loads(line)
                    if recipe.get("tokens"):
                        yield recipe["tokens"]
                except json.JSONDecodeError:
                    continue


def train_and_load(parsed_filepath):
    print("1. Initializing Streaming Corpus...")
    corpus = StreamRecipeCorpus(parsed_filepath)

    optimal_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"2. Training Word2Vec Model (Workers: {optimal_workers})...")

    model = Word2Vec(
        sentences=corpus,
        vector_size=300,
        window=20,
        min_count=15,
        sg=1,
        workers=optimal_workers,
        epochs=10,
    )

    vocab_size = len(model.wv.index_to_key)
    print(f"Training Complete! Vocabulary size: {vocab_size} unique ingredient states.")

    print("3. Connecting to PostgreSQL to bulk-load vectors...")
    try:
        conn = psycopg2.connect(DB_DSN)
        cursor = conn.cursor()

        print("4. Truncating old vectors from the database...")
        cursor.execute("TRUNCATE TABLE ingredient_vectors;")

        insert_data = []
        for token in model.wv.index_to_key:
            vector = model.wv[token].tolist()
            insert_data.append((token, str(vector)))

        insert_query = """
            INSERT INTO ingredient_vectors (token, embedding) 
            VALUES %s 
            ON CONFLICT (token) DO UPDATE SET embedding = EXCLUDED.embedding;
        """

        print(f"5. Executing bulk insert of {vocab_size} vectors...")
        execute_values(cursor, insert_query, insert_data, page_size=5000)

        conn.commit()
        print("Success! Vectors are securely indexed in PostgreSQL.")

    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        if "conn" in locals() and conn:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    input_file = "parsed_recipes.jsonl"
    train_and_load(input_file)
