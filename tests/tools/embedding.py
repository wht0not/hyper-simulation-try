import random

import numpy as np
import torch

_DEFAULT_SEED = 42
random.seed(_DEFAULT_SEED)
np.random.seed(_DEFAULT_SEED)
torch.manual_seed(_DEFAULT_SEED)
torch.cuda.manual_seed_all(_DEFAULT_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
_model_cache = {}
from sentence_transformers import SentenceTransformer

def _get_sentence_transformer() -> SentenceTransformer:
    if "Qwen/Qwen3-Embedding-0.6B" not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
        # model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")
        model = SentenceTransformer(local_model_path)
        model.eval()
        _model_cache["Qwen/Qwen3-Embedding-0.6B"] = model
    return _model_cache["Qwen/Qwen3-Embedding-0.6B"]

def get_embedding_batch(texts: list[str], batch_size: int=256, cache: None | dict[str, np.ndarray]=None) -> list[np.ndarray]:
    model = _get_sentence_transformer()
    
    if cache is None:
        cache = {}
        
    unique_texts = list(set(texts))
    missing_texts = [t for t in unique_texts if t not in cache]

    if missing_texts:
        new_embeddings = model.encode(
            missing_texts,
            batch_size=batch_size, # 让模型内部去处理分批
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        cache.update(zip(missing_texts, new_embeddings))

    return [cache[t] for t in texts]

PAIRS = [
    (
        "Q1",
        "What military activities were conducted by U.S. Forces and Japanese Self-Defense Forces in the Taiwan Strait during the Spring Festival of 2026?",
        [
            (
                "D1",
                "During the National Day of 2025, U.S. Military and Japanese Self-Defense Forces held the 'Keen Sword' joint military exercise in the Taiwan Strait.",
            )
        ],
    ),
    (
        "Q2",
        "The USS Gerald R. Ford aircraft carrier has been deployed to the northern Red Sea to conduct combat readiness missions.",
        [
            (
                "D2",
                "The USS Gerald R. Ford aircraft carrier has returned to Norfolk Naval Station for repairs and docking due to a fire.",
            )
        ],
    ),
    (
        "Q3",
        "The U.S. Supreme Court stopped collecting China-related tariffs imposed under the International Emergency Economic Powers Act after its ruling. It also halted the post-ruling enforcement of those China-related measures.",
        [
            (
                "D3",
                "The U.S. continues to impose an additional 10% import tariff on Chinese goods under Section 122 of the Trade Act of 1974. The government is enforcing that trade measure on covered imports.",
            )
        ],
    ),
    (
        "Q4",
        "Who is the highest military commander of the U.S. Air Force?",
        [
            (
                "D4",
                "General Charles Q. Brown once served as Chief of Staff of the U.S. Air Force, and later became Chairman of the Joint Chiefs of Staff.",
            ),
            (
                "D5",
                "The highest leader of the Department of the U.S. Air Force is Troy Meink, who is responsible for the organization, training, and equipping of the U.S. Air Force.",
            ),
            (
                "D6",
                "General Kenneth Wilsbach delivered a keynote speech as Chief of Staff of the U.S. Air Force at a 2026 warfare symposium.",
            ),
        ],
    ),
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def get_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return cosine_similarity(emb1, emb2)


def compute_pair_scores() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_id, query_text, docs in PAIRS:
        for doc_id, doc_text in docs:
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "similarity": round(get_similarity(query_text, doc_text), 6),
                    "query": query_text,
                    "doc": doc_text,
                }
            )
    return rows


def main() -> None:
    rows = compute_pair_scores()
    for row in rows:
        print(
            f'{row["query_id"]} vs {row["doc_id"]}: {row["similarity"]:.6f}'
        )


if __name__ == "__main__":
    main()
