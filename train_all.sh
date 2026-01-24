#!/bin/bash

uv run -m src.train --type text --tokenizer bpe
uv run -m src.train --type dna  --tokenizer bpe
uv run -m src.train --type dna  --tokenizer ovl