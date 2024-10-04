import os
import csv
from gensim.models import FastText, Word2Vec

def load_embedding_model(model_path):
    if 'fasttext' in model_path.lower():
        return FastText.load(model_path)
    else:
        return Word2Vec.load(model_path)

def get_vocabulary_vectors(model):
    vocab = model.wv.key_to_index
    vectors = {word: model.wv[word] for word in vocab}
    return vectors

def save_vectors_to_csv(vectors, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word'] + [f'dim_{i}' for i in range(len(next(iter(vectors.values()))))])
        for word, vector in vectors.items():
            writer.writerow([word] + vector.tolist())

def extract_embeddings(model_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(model_dir):
        if filename.endswith('.model'):
            model_path = os.path.join(model_dir, filename)
            print(f"Loading model: {filename}")
            model = load_embedding_model(model_path)

            print("Extracting vocabulary vectors...")
            vectors = get_vocabulary_vectors(model)

            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_vectors.csv")
            print(f"Saving vectors to: {output_file}")
            save_vectors_to_csv(vectors, output_file)

            print(f"Completed processing for {filename}")

def extract_embedding_vectors(model_dir, output_dir):
    print(f"Starting embedding extraction from {model_dir}")
    extract_embeddings(model_dir, output_dir)
    print("Embedding extraction completed.")