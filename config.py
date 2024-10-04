CONFIG = {
    'raw_data_dir': './dataset/raw/',
    'reports_dir': './dataset/reports/benign/',
    'processed_dir': './dataset/processed/',
    'extracted_dir': './dataset/extracted/',
    'features_dir': './dataset/extracted/features/',
    'embedding_dir': './dataset/models/embedding_models/',
    'processed_embedding_dir': './dataset/processed/embedding_vectors/',
    'compressed_embedding_dir': './dataset/processed/compressed_embedding_vectors/',
    'figure_dir': './dataset/figure/vector_range/',

    # 워드 임베딩 기본 설정
    'default_embedding_model': 'word2vec',  # 'word2vec', 'fasttext', 또는 'glove'
    'embedding_size': 100,
    'window_size': 5,
    'min_count': 1,
    'workers': 4,

    # Word2Vec 특정 설정
    'word2vec': {
        'sg': 1,  # 0 for CBOW, 1 for Skip-gram
    },

    # FastText 특정 설정
    'fasttext': {
        'sg': 1,  # 0 for CBOW, 1 for Skip-gram
    },

    # GloVe 특정 설정
    'glove': {
        'epochs': 25,
        'learning_rate': 0.05,
    },

    # 딥러닝 모델 설정
    'max_sequence_length': 1000,
    'batch_size': 32,
    'epochs': 10,
}