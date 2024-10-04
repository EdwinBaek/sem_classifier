import numpy as np

def quantize(value, min_val=-1.0, max_val=1.0, bits=8):
    """실수를 8비트 부호 있는 정수로 양자화"""
    range = max_val - min_val
    step = range / (2 ** bits - 1)
    return np.clip(np.round((value - min_val) / step) - 2 ** (bits - 1), -2 ** (bits - 1), 2 ** (bits - 1) - 1).astype(
        int)


def dequantize(value, min_val=-1.0, max_val=1.0, bits=8):
    """8비트 부호 있는 정수를 실수로 역양자화"""
    range = max_val - min_val
    step = range / (2 ** bits - 1)
    return (value + 2 ** (bits - 1)) * step + min_val


def get_frequency(data):
    """데이터의 빈도수 계산"""
    unique, counts = np.unique(data, return_counts=True)
    return dict(zip(unique, counts))


def get_cumulative_frequency(freq):
    """누적 빈도수 계산"""
    cf = {}
    total = 0
    for symbol, count in sorted(freq.items()):
        cf[symbol] = total
        total += count
    return cf, total


def arithmetic_encode(data):
    """Arithmetic coding을 사용한 인코딩"""
    freq = get_frequency(data)
    cf, total = get_cumulative_frequency(freq)

    low, high = 0.0, 1.0
    for symbol in data:
        range_width = high - low
        high = low + range_width * (cf[symbol] + freq[symbol]) / total
        low = low + range_width * cf[symbol] / total

    return (low + high) / 2


def arithmetic_decode(encoded_value, length, freq):
    """Arithmetic coding을 사용한 디코딩"""
    cf, total = get_cumulative_frequency(freq)
    decoded = []
    value = encoded_value

    for _ in range(length):
        for symbol, count in freq.items():
            if cf[symbol] <= value * total < cf[symbol] + count:
                decoded.append(symbol)
                range_width = (cf[symbol] + count - cf[symbol]) / total
                value = (value - cf[symbol] / total) / range_width
                break

    return np.array(decoded)


def create_codebook(vectors):
    """벡터 리스트로부터 코드북 생성"""
    codebook = {}
    for i, vec in enumerate(vectors):
        quantized = quantize(vec)
        encoded = arithmetic_encode(quantized)
        codebook[i] = {
            'original': vec,
            'quantized': quantized,
            'encoded': encoded,
            'freq': get_frequency(quantized)
        }
    return codebook