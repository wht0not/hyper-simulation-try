from __future__ import annotations

import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
INPUT_DIR = REPO_ROOT / "tests" / "data-utility" / "data" / "hypergraph" / "q3d3"
LOG_DIR = REPO_ROOT / "tests" / "data-utility" / "2-process" / "log"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hyper_simulation.component.denial as denial_module
import hyper_simulation.component.hyper_simulation as hs_module
import hyper_simulation.component.semantic_cluster as semantic_cluster_module
import hyper_simulation.utils.log as log_module
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph


def _build_plain_logger(logger_name: str, file_path: Path, level: str = "INFO") -> logging.Logger:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_map.get(level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(level_map.get(level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class RoutedLogger:
    def __init__(self, name: str, level: str = "INFO") -> None:
        self.name = name
        self.level = level

    def _pick_logger(self, message: str) -> logging.Logger:
        text = str(message)
        if self.name == "denial_comment":
            return _build_plain_logger("routed.hv", LOG_DIR / "hv.log", self.level)
        if self.name == "semantic_cluster":
            if "D-Match" in text or "Batch match exception" in text:
                return _build_plain_logger("routed.d-match", LOG_DIR / "d-match.log", self.level)
            return _build_plain_logger(
                "routed.hyper-cluster-pairs",
                LOG_DIR / "hyper cluster pairs.log",
                self.level,
            )
        if self.name == "hyper_simulation":
            return _build_plain_logger("routed.hyper-simulation", LOG_DIR / "hyper_simulation.log", self.level)
        return _build_plain_logger(f"routed.{self.name}", LOG_DIR / f"{self.name}.log", self.level)

    def debug(self, message: str, *args, **kwargs) -> None:
        self._pick_logger(message).debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        self._pick_logger(message).info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._pick_logger(message).warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        self._pick_logger(message).error(message, *args, **kwargs)


def routed_logger(name: str, level: str = "INFO", _log_dir: str = "logs") -> RoutedLogger:
    _ = _log_dir
    return RoutedLogger(name=name, level=level)


def patch_loggers() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Force the relevant modules to print INFO logs to stdout and also persist
    # the routed logs under tests/data-utility/2-process/log.
    log_module.getLogger = routed_logger
    hs_module.getLogger = routed_logger
    denial_module.getLogger = routed_logger
    semantic_cluster_module.getLogger = routed_logger


def load_hypergraphs() -> tuple[LocalHypergraph, LocalHypergraph]:
    query_path = INPUT_DIR / "query_hypergraph.pkl"
    data_path = INPUT_DIR / "data_hypergraph.pkl"
    if not query_path.exists():
        raise FileNotFoundError(f"Query hypergraph not found: {query_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data hypergraph not found: {data_path}")
    return LocalHypergraph.load(str(query_path)), LocalHypergraph.load(str(data_path))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    patch_loggers()

    query_hg, data_hg = load_hypergraphs()

    print("=== Begin Hyper Simulation ===")

    mapping, q_map, d_map = hs_module.compute_hyper_simulation(query_hg, data_hg)

    print()
    print("=== Final Mapping (returned value) ===")
    for q_id, d_ids in sorted(mapping.items()):
        q_text = q_map[q_id].text() if q_id in q_map else f"Q{q_id}"
        if d_ids:
            targets = ", ".join(
                f"D{d_id}:'{d_map[d_id].text()}'" if d_id in d_map else f"D{d_id}"
                for d_id in sorted(d_ids)
            )
        else:
            targets = "-"
        print(f"Q{q_id} '{q_text}' -> {targets}")


if __name__ == "__main__":
    main()
