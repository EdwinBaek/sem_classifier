import os
import csv
import numpy as np
import struct
from collections import Counter

def quantize_vector(vector, compression_bits=8):
    min_val, max_val = np.min(vector), np.max(vector)
    step = (max_val - min_val) / (2 ** compression_bits - 1)
    quantized = np.round((vector - min_val) / step).astype(int)
    return quantized, min_val, max_val

def get_probabilities(data):
    counts = Counter(data)
    total = sum(counts.values())
    return {symbol: count / total for symbol, count in counts.items()}

def get_cumulative_probs(probabilities):
    cumulative = {}
    total = 0
    for symbol, prob in sorted(probabilities.items()):
        cumulative[symbol] = (total, total + prob)
        total += prob
    return cumulative

def arithmetic_encode(data, probabilities):
    cumulative_probs = get_cumulative_probs(probabilities)
    low, high = 0.0, 1.0
    for symbol in data:
        range_width = high - low
        high = low + range_width * cumulative_probs[symbol][1]
        low = low + range_width * cumulative_probs[symbol][0]
    return (low + high) / 2

def compress_vector(vector, compression_bits=8):
    quantized, min_val, max_val = quantize_vector(vector, compression_bits)
    probabilities = get_probabilities(quantized)
    encoded = arithmetic_encode(quantized, probabilities)
    encoded_bytes = struct.pack('!d', encoded)

    return {
        'encoded_bytes': encoded_bytes,
        'min_val': min_val,
        'max_val': max_val,
        'probabilities': probabilities
    }

def process_csv_file(input_file, output_file, compression_bits=8):
    compressed_vectors = {}
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word']
            vector = np.array([float(row[f'dim_{i}']) for i in range(128)])  # Assuming 128 dimensions
            compressed_vectors[word] = compress_vector(vector, compression_bits)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'encoded_bytes', 'min_val', 'max_val', 'probabilities'])
        for word, compressed_data in compressed_vectors.items():
            writer.writerow([
                word,
                compressed_data['encoded_bytes'].hex(),
                compressed_data['min_val'],
                compressed_data['max_val'],
                repr(compressed_data['probabilities'])
            ])

def compress_embedding_vectors(input_dir, output_dir, coding_type='arithmetic_coding', compression_bits=8):
    print(f"Starting embedding compression from {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"compressed_{filename}")
            print(f"Processing file: {filename}")
            process_csv_file(input_file, output_file, compression_bits)
            print(f"Compressed file saved as: {output_file}")

    print("Embedding compression completed.")