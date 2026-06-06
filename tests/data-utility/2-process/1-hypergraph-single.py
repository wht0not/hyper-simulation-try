from __future__ import annotations

import logging
import sys
from pathlib import Path

q3 = "The U.S. Supreme Court stopped collecting China-related tariffs imposed under the International Emergency Economic Powers Act after its ruling. It also halted the post-ruling enforcement of those China-related measures."
d3 = "The United States continues to impose an additional 10%_import_tariff on Chinese goods under Section 122 of the Trade Act of 1974. The government is still enforcing that trade measure on covered imports."

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
LOCOMO_ROOT = REPO_ROOT / "tests" / "tasks" / "memory" / "locomo"
OUTPUT_DIR = REPO_ROOT / "tests" / "data-utility" / "data" / "hypergraph" / "q3d3"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(LOCOMO_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCOMO_ROOT))

from method.hyper_simulation.build import batch_text_to_hypergraph, setup_gpu_nlp


def _build_one(text: str, *, is_query: bool, nlp):
    payload = [{"text": text, "meta": {"label": "query" if is_query else "data"}}]
    meta, hypergraph = list(
        batch_text_to_hypergraph(
            nlp=nlp,
            texts_with_metadata=payload,
            batch_size=1,
            is_query=is_query,
        )
    )[0]
    if hypergraph is None:
        raise RuntimeError(f"Failed to build hypergraph: {meta.get('error', 'unknown error')}")
    return hypergraph


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Reuse the same build entry points as the LoCoMo hypergraph build pipeline.
    nlp = setup_gpu_nlp()
    q_hg = _build_one(q3, is_query=True, nlp=nlp)
    d_hg = _build_one(d3, is_query=False, nlp=nlp)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    q_path = OUTPUT_DIR / "query_hypergraph.pkl"
    d_path = OUTPUT_DIR / "data_hypergraph.pkl"
    q_hg.save(str(q_path))
    d_hg.save(str(d_path))

    print("=== Query Hypergraph (q3) ===")
    print(q_hg)
    print()
    print("=== Data Hypergraph (d3) ===")
    print(d_hg)
    print()
    print(f"Saved query hypergraph to: {q_path}")
    print(f"Saved data hypergraph to: {d_path}")


if __name__ == "__main__":
    main()
