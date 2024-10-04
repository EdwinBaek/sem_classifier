import os
import csv
from collections import defaultdict
from gensim.models import Word2Vec, FastText


def load_features(input_dir):
    features = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in os.walk(input_dir):
        feature_type = os.path.basename(root)
        if feature_type.startswith(('dynamic-', 'static-')):
            for file in files:
                if file.endswith('.csv'):
                    file_hash = os.path.splitext(file)[0]  # Get file name without extension
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        features[feature_type][file_hash] = [row[0] for row in reader]
    return features


def train_word2vec(sentences, feature_name, save_dir):
    model = Word2Vec(
        vector_size=128, window=5, min_count=1, workers=4, sg=1, hs=0, negative=10,
        ns_exponent=0.75, cbow_mean=0, alpha=0.025, sample=1e-4, shrink_windows=True
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=100, compute_loss=True)
    save_model = os.path.join(save_dir, f"word2vec_{feature_name}.model")
    model.save(save_model)

def train_glove_like(sentences, feature_name, save_dir):
    model = Word2Vec(
        vector_size=128, window=5, min_count=1, workers=4, sg=1, hs=0, negative=10,
        ns_exponent=0.75, alpha=0.025, min_alpha=0.0001, sample=1e-4
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=100, compute_loss=True)
    save_model = os.path.join(save_dir, f"glove_like_{feature_name}.model")
    model.save(save_model)

def train_fasttext(sentences, feature_name, save_dir):
    model = FastText(
        vector_size=128, window=5, min_count=1, workers=4, sg=1, hs=0, negative=10,
        ns_exponent=0.75, alpha=0.025, min_alpha=0.0001, sample=1e-4, min_n=2,max_n=5, word_ngrams=1, bucket=2000000
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=100)
    save_model = os.path.join(save_dir, f"fasttext_{feature_name}.model")
    model.save(save_model)

def train_embedding_models(input_dir, output_dir):
    print("Extracted features directory : %s" % input_dir)
    print("Embedding model save directory : %s" % output_dir)
    print("Loading features...")
    features = load_features(input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Training word embedding models...")
    total_features = len(features)
    for i, (feature_name, feature_data) in enumerate(features.items(), 1):
        print(f"\nTraining models for {feature_name} ({i}/{total_features})")

        # Convert feature data to sentences (list of lists)
        sentences = [feature_data[file_hash] for file_hash in feature_data]

        # Word2Vec
        print("Training Word2Vec model...")
        try:
            train_word2vec(sentences, feature_name, output_dir)
        except Exception as e:
            print(str(e))
            pass

        # GloVe-like (using Word2Vec)
        print("Training GloVe-like model...")
        try:
            train_glove_like(sentences, feature_name, output_dir)
        except Exception as e:
            print(str(e))
            pass

        # FastText
        print("Training FastText model...")
        try:
            train_fasttext(sentences, feature_name, output_dir)
        except Exception as e:
            print(str(e))
            pass

    print("Word embedding models training completed.")