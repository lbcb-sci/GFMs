from transformers import AutoTokenizer

bpe = AutoTokenizer.from_pretrained('runs/90M_dna_bpe/1', local_files_only=True)
kmer = AutoTokenizer.from_pretrained('runs/90M_dna_kmer/1', local_files_only=True)

vocab_size = bpe.vocab_size

vocab_bpe = set(bpe.get_vocab().keys())
vocab_kmer = set(kmer.get_vocab().keys())

inter = vocab_bpe & vocab_kmer
print(f'bpe and k-mer tokenizer are {len(inter) / vocab_size * 100:.2f}% similar')
print(f'they have {len(inter)} / {vocab_size} tokens in common')
