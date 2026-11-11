# SPDX-License-Identifier: Apache-2.0
"""A tiny random CTC ASR model, built locally, for tests that need *an* ASR.

Stands in for the ~1 GB default where transcript content is not under test:
it loads through the same ``transformers`` ASR pipeline (type ``ctc``), so the
whole cascade runs, but needs no download and is a few hundred KB. Its
transcripts are noise, often empty.
"""

import json
from pathlib import Path

_VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "a": 5, "b": 6}


def build_tiny_asr(path) -> Path:
    """Write a loadable random Wav2Vec2ForCTC repo to ``path`` and return it."""
    import torch
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.json").write_text(json.dumps(_VOCAB))
    tokenizer = Wav2Vec2CTCTokenizer(
        str(path / "vocab.json"), pad_token="<pad>", word_delimiter_token="|"
    )
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True
    )
    Wav2Vec2Processor(feature_extractor=extractor, tokenizer=tokenizer).save_pretrained(
        path
    )
    config = Wav2Vec2Config(
        vocab_size=len(_VOCAB),
        pad_token_id=0,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(16, 16),
        conv_stride=(5, 4),
        conv_kernel=(10, 4),
        num_feat_extract_layers=2,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    Wav2Vec2ForCTC(config).save_pretrained(path, safe_serialization=True)
    return path
