'''
import numpy as np

mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def kmer_to_index(kmer: str) -> int:
    idx = 0
    for ch in kmer:
        idx = idx * 4 + mapping[ch]
    return idx

def markov_transition_matrix(sequence: str, kmer: int, laplace: bool = True):
    counts = np.zeros(shape=(4**kmer, 4), dtype=float)

    N = len(sequence)

    for index in range(0, N-kmer):
        s = sequence[index:index+kmer]
        y = sequence[index+kmer]
        counts[kmer_to_index(s), mapping[y]] += 1

    if laplace: counts += 1.0

    rowsums = np.sum(counts, axis=1, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        tmat = counts / rowsums
        tmat[~np.isfinite(tmat)] = 0.0

    return tmat

def _kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))

def markov_js_distance(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape
    n = P.shape[0]

    d_rows = []
    for i in range(n):
        p = P[i]
        q = Q[i]
        m = 0.5 * (p + q)
        js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        d_rows.append(np.sqrt(js))  # Jensen–Shannon distance

    return float(np.mean(d_rows))

def markov_distance(seqA, seqB, kmer: int):
    t1 = markov_transition_matrix(seqA, kmer)
    t2 = markov_transition_matrix(seqB, kmer)
    return markov_js_distance(t1, t2)
'''

import numpy as np

_char_to_int = np.full(256, -1, dtype=np.int8)
_char_to_int[ord('A')] = 0
_char_to_int[ord('C')] = 1
_char_to_int[ord('G')] = 2
_char_to_int[ord('T')] = 3

def encode_seq(sequence: str) -> np.ndarray:
    arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
    enc = _char_to_int[arr]
    return enc.astype(np.int8)

def markov_transition_matrix(sequence: str, kmer: int, laplace: bool = True):
    seq = encode_seq(sequence)
    n = len(seq)

    num_states = 4 ** kmer
    counts = np.zeros((num_states, 4), dtype=float)

    # Rolling k-mer index: base-4 number
    idx = 0
    base = 4 ** (kmer - 1)

    # Initialize with first kmer
    for i in range(kmer):
        idx = idx * 4 + seq[i]

    # First transition
    counts[idx, seq[kmer]] += 1

    # Slide window over sequence
    for i in range(1, n - kmer):
        # remove highest base digit and add new one
        idx = (idx % base) * 4 + seq[i + kmer]
        counts[idx, seq[i + kmer]] += 1

    if laplace:
        counts += 1.0

    rowsums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        tmat = counts / rowsums
        tmat[~np.isfinite(tmat)] = 0.0
    return tmat

def _kl_rows(P, Q, eps=1e-12):
    """Row-wise KL(P||Q) for 2D arrays P, Q with same shape."""
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return np.sum(P * np.log(P / Q), axis=1)

def markov_js_distance(P, Q):
    M = 0.5 * (P + Q)
    kl_pm = _kl_rows(P, M)
    kl_qm = _kl_rows(Q, M)
    js = 0.5 * (kl_pm + kl_qm)
    d = np.sqrt(js)
    return float(d.mean())

def markov_distance(seqA, seqB, kmer: int):
    t1 = markov_transition_matrix(seqA, kmer)
    t2 = markov_transition_matrix(seqB, kmer)
    return markov_js_distance(t1, t2)
