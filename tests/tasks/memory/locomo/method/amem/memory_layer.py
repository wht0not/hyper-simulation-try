from __future__ import annotations

import json
import importlib
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


def simple_tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"\W+", str(text).lower()) if tok]


def _simple_keywords(text: str, top_k: int = 8) -> list[str]:
    words = simple_tokenize(text)
    if not words:
        return []
    freq: dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:top_k]]


def _safe_json_loads(raw: str, default: dict[str, Any]) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return default
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if not text.startswith("{"):
        start_idx = text.find("{")
        if start_idx >= 0:
            text = text[start_idx:]
    if not text.endswith("}"):
        end_idx = text.rfind("}")
        if end_idx >= 0:
            text = text[: end_idx + 1]
    try:
        parsed = json.loads(text)
    except Exception:
        return default
    if not isinstance(parsed, dict):
        return default
    return parsed


_EMBEDDING_CACHE: dict[str, Any] = {}


def _get_ollama_embeddings(model_name: str = "qwen3-embedding:0.6b"):
    if model_name not in _EMBEDDING_CACHE:
        from langchain_ollama import OllamaEmbeddings

        _EMBEDDING_CACHE[model_name] = OllamaEmbeddings(model=model_name)
    return _EMBEDDING_CACHE[model_name]


class OllamaController:
    def __init__(self, model: str = "qwen3.5:9b"):
        from langchain_ollama import ChatOllama
        self._llm = ChatOllama(model=model, temperature=0.0, reasoning=False)

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        _ = temperature
        response = self._llm.invoke(prompt)
        return str(getattr(response, "content", "") or "").strip()


class LLMController:
    # only ollama, no other backends
    def __init__(self, model: str = "qwen3.5:9b"):
        self.llm = OllamaController(model=model)


class MemoryNote:
    """A-mem-like memory unit with metadata and optional LLM extraction."""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        links: Optional[list[int]] = None,
        importance_score: Optional[float] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        evolution_history: Optional[list[Any]] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        llm_controller: Optional[LLMController] = None,
    ):
        self.content = str(content or "")
        self.id = id or str(uuid.uuid4())
        self.links = list(links or [])
        self.importance_score = float(importance_score or 1.0)
        self.retrieval_count = int(retrieval_count or 0)
        now = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or now
        self.last_accessed = last_accessed or now
        self.evolution_history = list(evolution_history or [])
        self.category = category or "Uncategorized"

        if llm_controller and any(param is None for param in [keywords, context, tags]):
            analysis = self.analyze_content(self.content, llm_controller)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.keywords = list(keywords or _simple_keywords(self.content))
        self.context = str(context or "General")
        self.tags = list(tags or [])

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> dict[str, Any]:
        prompt = f"""
Generate a JSON object with fields: keywords (string array), context (string), tags (string array).
Keep it concise and grounded in the content.

Content:
{content}
"""
        default = {"keywords": _simple_keywords(content), "context": "General", "tags": []}
        try:
            response = llm_controller.llm.get_completion(prompt, temperature=0.3)
            parsed = _safe_json_loads(response, default=default)
            keywords = parsed.get("keywords", default["keywords"])
            context = parsed.get("context", default["context"])
            tags = parsed.get("tags", default["tags"])
            if not isinstance(keywords, list):
                keywords = default["keywords"]
            if not isinstance(tags, list):
                tags = default["tags"]
            return {
                "keywords": [str(one) for one in keywords if str(one).strip()],
                "context": str(context or "General"),
                "tags": [str(one) for one in tags if str(one).strip()],
            }
        except Exception:
            return default


class SimpleEmbeddingRetriever:
    """Simple retriever using only sentence-transformer embeddings."""

    def __init__(self, model_name: str = "qwen3-embedding:0.6b"):
        self._emb = _get_ollama_embeddings(model_name=model_name)
        self.corpus: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.document_ids: dict[str, int] = {}

    def add_documents(self, documents: list[str]) -> None:
        docs = [str(doc).strip() for doc in documents if str(doc).strip()]
        if not docs:
            return
        vectors = np.array(self._emb.embed_documents(docs), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        start_idx = len(self.corpus)
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.corpus.extend(docs)
        for idx, doc in enumerate(docs):
            self.document_ids[doc] = start_idx + idx

    def add_documents_with_embeddings(self, documents: list[str], embeddings: list[list[float]]) -> None:
        docs = [str(doc).strip() for doc in documents if str(doc).strip()]
        if not docs or len(docs) != len(embeddings):
            return
        vectors = np.array(embeddings, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(docs):
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        start_idx = len(self.corpus)
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.corpus.extend(docs)
        for idx, doc in enumerate(docs):
            self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> list[int]:
        if not self.corpus or self.embeddings is None:
            return []
        query_vec = np.array(self._emb.embed_query(str(query)), dtype=float)
        query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
        denom = np.linalg.norm(self.embeddings, axis=1) * max(np.linalg.norm(query_vec), 1e-12)
        scores = np.dot(self.embeddings, query_vec) / np.maximum(denom, 1e-12)
        top_k = max(1, min(int(k), len(self.corpus)))
        return np.argsort(scores)[-top_k:][::-1].tolist()

    @classmethod
    def load_from_local_memory(cls, memories: dict[str, MemoryNote], model_name: str) -> "SimpleEmbeddingRetriever":
        retriever = cls(model_name=model_name)
        docs = []
        for memory in memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            docs.append(f"{memory.content} , {metadata_text}")
        retriever.add_documents(docs)
        return retriever


class HybridRetriever:
    """
    Hybrid retrieval:
    - semantic: sentence-transformer embeddings
    - lexical: BM25 (if available), otherwise token-overlap
    """

    def __init__(self, alpha: float = 0.65, model_name: str = "qwen3-embedding:0.6b"):
        self._emb = _get_ollama_embeddings(model_name=model_name)
        self.alpha = float(alpha)
        self.corpus: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._tokenized_docs: list[list[str]] = []
        self._bm25_cls = None
        self._bm25 = None
        self.document_ids: dict[str, int] = {}
        try:
            module = importlib.import_module("rank_bm25")
            self._bm25_cls = getattr(module, "BM25Okapi", None)
        except Exception:
            self._bm25_cls = None

    def _rebuild_lexical(self) -> None:
        if self._bm25_cls is not None and self._tokenized_docs:
            self._bm25 = self._bm25_cls(self._tokenized_docs)
        else:
            self._bm25 = None

    def add_documents(self, documents: list[str]) -> None:
        docs = [str(doc).strip() for doc in documents if str(doc).strip()]
        if not docs:
            return
        vectors = np.array(self._emb.embed_documents(docs), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        start_idx = len(self.corpus)
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.corpus.extend(docs)
        self._tokenized_docs.extend([simple_tokenize(doc) for doc in docs])
        for idx, doc in enumerate(docs):
            self.document_ids[doc] = start_idx + idx
        self._rebuild_lexical()

    def add_documents_with_embeddings(self, documents: list[str], embeddings: list[list[float]]) -> None:
        docs = [str(doc).strip() for doc in documents if str(doc).strip()]
        if not docs or len(docs) != len(embeddings):
            return
        vectors = np.array(embeddings, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(docs):
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        start_idx = len(self.corpus)
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        self.corpus.extend(docs)
        self._tokenized_docs.extend([simple_tokenize(doc) for doc in docs])
        for idx, doc in enumerate(docs):
            self.document_ids[doc] = start_idx + idx
        self._rebuild_lexical()

    def _lexical_scores(self, query_tokens: list[str]) -> np.ndarray:
        if not self.corpus:
            return np.array([], dtype=float)
        if self._bm25 is not None:
            return np.array(self._bm25.get_scores(query_tokens), dtype=float)
        # fallback token-overlap score
        scores = []
        query_set = set(query_tokens)
        for doc_tokens in self._tokenized_docs:
            if not doc_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_set.intersection(set(doc_tokens)))
            scores.append(float(overlap) / max(len(query_set), 1))
        return np.array(scores, dtype=float)

    def search(self, query: str, k: int = 5) -> list[int]:
        if not self.corpus or self.embeddings is None:
            return []

        query_text = str(query)
        query_vec = np.array(self._emb.embed_query(query_text), dtype=float)
        query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-12)
        semantic_scores = np.dot(self.embeddings, query_vec)
        lexical_scores = self._lexical_scores(simple_tokenize(query_text))

        if lexical_scores.size:
            lexical_scores = (
                lexical_scores - lexical_scores.min()
            ) / (lexical_scores.max() - lexical_scores.min() + 1e-12)
        if semantic_scores.size:
            semantic_scores = (
                semantic_scores - semantic_scores.min()
            ) / (semantic_scores.max() - semantic_scores.min() + 1e-12)

        hybrid = self.alpha * semantic_scores + (1.0 - self.alpha) * lexical_scores
        top_k = max(1, min(int(k), len(self.corpus)))
        return np.argsort(hybrid)[-top_k:][::-1].tolist()


class AgenticMemorySystem:
    """
    High-fidelity local A-mem:
    - keep original class/interface names
    - keep add_note/process_memory/find_related_memories(_raw) flow
    - avoid fragile external dependency chain
    """

    def __init__(
        self,
        model_name: str = "qwen3-embedding:0.6b",
        llm_backend: str = "ollama",
        llm_model: str = "qwen3.5:9b",
        evo_threshold: int = 100,
        output_dir: str | None = None,
        namespace: str | None = None,
    ):
        if llm_backend != "ollama":
            raise ValueError("locomo/amem only supports llm_backend='ollama'.")
        self._embedding_model_name = model_name
        self.memories: dict[str, MemoryNote] = {}
        self.retriever = HybridRetriever(model_name=self._embedding_model_name)
        self.llm_controller = LLMController(model=llm_model)
        self.evo_threshold = int(evo_threshold)
        self.evo_cnt = 0
        self._namespace = str(namespace or "default").strip() or "default"
        self._memory_dir = Path(output_dir) / "memory" if output_dir else None
        if self._memory_dir is not None:
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            self._load_memories_from_disk()
        self.evolution_system_prompt = """
You are an AI memory evolution agent.
Return a JSON object with fields:
- should_evolve (boolean)
- actions (string array, choose from ["strengthen","update_neighbor"])
- suggested_connections (integer array)
- tags_to_update (string array)
- new_context_neighborhood (string array)
- new_tags_neighborhood (array of string arrays)

new memory:
context: {context}
content: {content}
keywords: {keywords}

nearest memories:
{nearest_neighbors_memories}
"""

    def _memory_file_path(self, memory_id: str) -> Path | None:
        if self._memory_dir is None:
            return None
        safe_ns = re.sub(r"[^a-zA-Z0-9._-]", "_", self._namespace)
        return self._memory_dir / f"amem_{safe_ns}.json"

    def _memory_document(self, note: MemoryNote) -> str:
        return (
            "content:"
            + note.content
            + " context:"
            + note.context
            + " keywords: "
            + ", ".join(note.keywords)
            + " tags: "
            + ", ".join(note.tags)
        )

    def _memory_payload(self, note: MemoryNote) -> dict[str, Any]:
        return {
            "memory_id": note.id,
            "method": "amem",
            "namespace": self._namespace,
            "content": note.content,
            "timestamp": note.timestamp,
            "metadata": {
                "context": note.context,
                "keywords": note.keywords,
                "tags": note.tags,
                "links": note.links,
            },
        }

    def _persist_note(self, note: MemoryNote) -> None:
        path = self._memory_file_path(note.id)
        if path is None:
            return
        payload = {
            "method": "amem",
            "namespace": self._namespace,
            "memory_id": self._namespace,
            "updated_at": datetime.now().strftime("%Y%m%d%H%M"),
            "memories": [self._memory_payload(one_note) for one_note in self.memories.values()],
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)

    def _load_memories_from_disk(self) -> None:
        if self._memory_dir is None:
            return
        memory_file = self._memory_file_path(self._namespace)
        if memory_file is None or not memory_file.exists():
            return
        try:
            payload = json.loads(memory_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        rows = payload.get("memories", [])
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            memory_id = str(row.get("memory_id", row.get("id", ""))).strip()
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            content = str(row.get("content", "")).strip()
            if not memory_id or not content:
                continue
            note = MemoryNote(
                content=content,
                id=memory_id,
                keywords=metadata.get("keywords", []),
                links=metadata.get("links", []),
                importance_score=1.0,
                retrieval_count=0,
                timestamp=str(row.get("timestamp", row.get("created_at", ""))).strip() or None,
                last_accessed=None,
                context=str(metadata.get("context", "General")),
                evolution_history=[],
                category="Uncategorized",
                tags=metadata.get("tags", []),
            )
            self.memories[note.id] = note
            self.retriever.add_documents([self._memory_document(note)])

    def add_note(self, content: str, time: str | None = None, **kwargs: Any) -> str:
        note_id = str(kwargs.get("id", "")).strip()
        if note_id and note_id in self.memories:
            return note_id
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        should_evolve, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([self._memory_document(note)])
        self._persist_note(note)
        if should_evolve:
            self.evo_cnt += 1
            if self.evo_threshold > 0 and self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self) -> None:
        self.retriever = HybridRetriever(model_name="qwen3-embedding:0.6b")
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def process_memory(self, note: MemoryNote) -> tuple[bool, MemoryNote]:
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        if not indices:
            return False, note

        prompt = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
        )
        default = {
            "should_evolve": False,
            "actions": [],
            "suggested_connections": [],
            "tags_to_update": [],
            "new_context_neighborhood": [],
            "new_tags_neighborhood": [],
        }
        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.2)
            decision = _safe_json_loads(response, default=default)
            should_evolve = bool(decision.get("should_evolve", False))
            if not should_evolve:
                return False, note

            actions = decision.get("actions", [])
            if "strengthen" in actions:
                connections = decision.get("suggested_connections", [])
                if isinstance(connections, list):
                    note.links.extend([int(x) for x in connections if str(x).isdigit()])
                tags_to_update = decision.get("tags_to_update", [])
                if isinstance(tags_to_update, list) and tags_to_update:
                    note.tags = [str(x) for x in tags_to_update if str(x).strip()]

            if "update_neighbor" in actions:
                notes_list = list(self.memories.values())
                notes_ids = list(self.memories.keys())
                new_contexts = decision.get("new_context_neighborhood", [])
                new_tags = decision.get("new_tags_neighborhood", [])
                if not isinstance(new_contexts, list):
                    new_contexts = []
                if not isinstance(new_tags, list):
                    new_tags = []
                for i in range(min(len(indices), len(notes_list))):
                    mem_idx = int(indices[i])
                    if mem_idx < 0 or mem_idx >= len(notes_list):
                        continue
                    n = notes_list[mem_idx]
                    if i < len(new_contexts) and str(new_contexts[i]).strip():
                        n.context = str(new_contexts[i]).strip()
                    if i < len(new_tags) and isinstance(new_tags[i], list):
                        n.tags = [str(tag) for tag in new_tags[i] if str(tag).strip()]
                    self.memories[notes_ids[mem_idx]] = n
            return True, note
        except Exception:
            return False, note

    def find_related_memories(self, query: str, k: int = 5) -> tuple[str, list[int]]:
        if not self.memories:
            return "", []
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            if i < 0 or i >= len(all_memories):
                continue
            memory_str += (
                "memory index:"
                + str(i)
                + "\t talk start time:"
                + all_memories[i].timestamp
                + "\t memory content: "
                + all_memories[i].content
                + "\t memory context: "
                + all_memories[i].context
                + "\t memory keywords: "
                + str(all_memories[i].keywords)
                + "\t memory tags: "
                + str(all_memories[i].tags)
                + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        if not self.memories:
            return ""
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            if i < 0 or i >= len(all_memories):
                continue
            memory_str += (
                "talk start time:"
                + all_memories[i].timestamp
                + "memory content: "
                + all_memories[i].content
                + "memory context: "
                + all_memories[i].context
                + "memory keywords: "
                + str(all_memories[i].keywords)
                + "memory tags: "
                + str(all_memories[i].tags)
                + "\n"
            )
            for j, neighbor in enumerate(all_memories[i].links):
                if j >= k:
                    break
                if int(neighbor) < 0 or int(neighbor) >= len(all_memories):
                    continue
                nb = all_memories[int(neighbor)]
                memory_str += (
                    "talk start time:"
                    + nb.timestamp
                    + "memory content: "
                    + nb.content
                    + "memory context: "
                    + nb.context
                    + "memory keywords: "
                    + str(nb.keywords)
                    + "memory tags: "
                    + str(nb.tags)
                    + "\n"
                )
        return memory_str
