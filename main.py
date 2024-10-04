import os
import argparse
from dotenv import load_dotenv
from src.models import embedding_model, statistical_coder
from src.training import embedding_trainer

# .env 파일 로드
load_dotenv(dotenv_path='.env')

if __name__ == "__main__":
    # 각 feature에 대한 embedding model 학습 (Word2Vec, GloVe, FastText)
    embedding_trainer.train_embedding_models(os.getenv('STATIC_FEATURE_DIR'), os.getenv('EMBEDDING_MODEL_DIR'))
    embedding_trainer.train_embedding_models(os.getenv('DYNAMIC_FEATURE_DIR'), os.getenv('EMBEDDING_MODEL_DIR'))

    # Load embedding model and save each embedding vectors
    embedding_model.extract_embedding_vectors(os.getenv('EMBEDDING_MODEL_DIR'), os.getenv('EMBEDDING_VECTOR_DIR'))

    # Load embedding vectors and generate codebook for each vocabulary
    statistical_coder.compress_embedding_vectors(os.getenv('PROCESSED_VECTOR_DIR'), os.getenv('ARITHMETIC_VECTOR_DIR'))