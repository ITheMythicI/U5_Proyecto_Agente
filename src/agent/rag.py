"""
Módulo RAG (Retrieval-Augmented Generation) del Agente Deportivo.
Implementa fragmentación de documentos, generación de embeddings densos,
indexación vectorial y recuperación semántica fundamentada en similitud de coseno.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .config import config

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Representa un fragmento indexado de conocimiento."""
    chunk_id: str
    source_file: str
    section_title: str
    text: str
    embedding: Optional[np.ndarray] = None


class DocumentProcessor:
    """Prepara y divide documentos Markdown en fragmentos semánticamente cohesivos."""

    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir

    def load_and_chunk_all(self) -> List[DocumentChunk]:
        """Carga todos los archivos Markdown y genera los chunks."""
        chunks: List[DocumentChunk] = []
        if not self.knowledge_dir.exists():
            return chunks

        md_files = sorted(list(self.knowledge_dir.glob("*.md")))
        for fpath in md_files:
            chunks.extend(self._chunk_markdown_file(fpath))
        return chunks

    def _chunk_markdown_file(self, fpath: Path) -> List[DocumentChunk]:
        """Divide un archivo Markdown respetando encabezados de sección."""
        text = fpath.read_text(encoding="utf-8")
        filename = fpath.name

        # Separar por encabezados ## o ###
        sections = re.split(r"\n(?=##?\s)", text)
        file_chunks: List[DocumentChunk] = []

        chunk_idx = 0
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            lines = sec.split("\n")
            title_line = lines[0].strip("# ").strip()
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else lines[0]

            if not body:
                body = title_line

            # Si la sección es muy larga (> 1200 caracteres), fragmentar por párrafos
            if len(body) > 1200:
                paragraphs = sec.split("\n\n")
                buffer = ""
                for p in paragraphs:
                    if len(buffer) + len(p) > 900:
                        if buffer:
                            chunk_idx += 1
                            file_chunks.append(DocumentChunk(
                                chunk_id=f"{filename}#{chunk_idx}",
                                source_file=filename,
                                section_title=title_line,
                                text=buffer.strip()
                            ))
                            buffer = p + "\n\n"
                    else:
                        buffer += p + "\n\n"
                if buffer.strip():
                    chunk_idx += 1
                    file_chunks.append(DocumentChunk(
                        chunk_id=f"{filename}#{chunk_idx}",
                        source_file=filename,
                        section_title=title_line,
                        text=buffer.strip()
                    ))
            else:
                chunk_idx += 1
                file_chunks.append(DocumentChunk(
                    chunk_id=f"{filename}#{chunk_idx}",
                    source_file=filename,
                    section_title=title_line,
                    text=sec
                ))

        return file_chunks


class DenseVectorStore:
    """
    Almacén vectorial con soporte explícito para dos modos independientes:
    1. Gemini Neural Embeddings (3072 dimensiones con modelo gemini-embedding-001).
    2. Local LSA (Análisis Semántico Latente: TF-IDF sublineal + TruncatedSVD a 32 dimensiones).
    """

    def __init__(self, embedding_dim: int = 32, mode: str = "auto"):
        self.embedding_dim = embedding_dim
        self.mode = mode  # 'auto', 'gemini', 'local_lsa'
        self.active_mode: str = "local_lsa"
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None  # Matriz N x D
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None

    def build_index(self, chunks: List[DocumentChunk], forced_mode: Optional[str] = None) -> None:
        """Genera embeddings e indexa la matriz vectorial en el modo seleccionado."""
        self.chunks = chunks
        if not chunks:
            self.vectors = np.empty((0, self.embedding_dim))
            return

        target_mode = forced_mode or self.mode
        corpus = [f"{c.section_title}. {c.text}" for c in chunks]

        # 1. Modo Gemini Neural Embeddings
        if target_mode in ["gemini", "auto"] and config.is_gemini_configured and GENAI_AVAILABLE:
            try:
                client = genai.Client(api_key=config.gemini_api_key)
                res = client.models.embed_content(
                    model=config.gemini_embedding_model,
                    contents=corpus
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    vec_list = [e.values for e in res.embeddings]
                    self.vectors = normalize(np.array(vec_list), norm="l2")
                    self.active_mode = "gemini_neural_3072d"
                    return
            except Exception:
                if target_mode == "gemini":
                    raise RuntimeError("Fallo al generar embeddings con Gemini y se forzó mode='gemini'.")
                # Si falló en modo 'auto', recurre a LSA local

        # 2. Modo Local LSA (Latent Semantic Analysis: TF-IDF + SVD)
        self.active_mode = "local_lsa_32d"
        spanish_stop_words = [
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
            "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
            "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
            "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra"
        ]

        n_components = min(self.embedding_dim, max(2, len(corpus) - 1))
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            ngram_range=(1, 2),
            stop_words=spanish_stop_words,
            max_features=2500
        )
        tfidf_mat = self.vectorizer.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense_vecs = self.svd.fit_transform(tfidf_mat)
        self.vectors = normalize(dense_vecs, norm="l2")

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recupera los top_k fragmentos más afines semánticamente mediante similitud de coseno."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        # Vectorizar consulta según el modo activo
        if self.active_mode.startswith("gemini"):
            try:
                client = genai.Client(api_key=config.gemini_api_key)
                res = client.models.embed_content(
                    model=config.gemini_embedding_model,
                    contents=query_text
                )
                q_vec = normalize(np.array([res.embeddings[0].values]), norm="l2")
            except Exception:
                return []
        else:
            if self.vectorizer and self.svd:
                tfidf_q = self.vectorizer.transform([query_text])
                dense_q = self.svd.transform(tfidf_q)
                q_vec = normalize(dense_q, norm="l2")
            else:
                return []

        # Similitud de coseno
        similarities = np.dot(self.vectors, q_vec.T).ravel()
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "similarity_score": round(score, 4),
                "embedding_mode": self.active_mode
            })

        return results


class RAGPipeline:
    """Orquestador completo del sistema RAG con soporte para modos Gemini y Local LSA."""

    def __init__(self, knowledge_dir: Optional[Path] = None, mode: str = "auto"):
        self.knowledge_dir = knowledge_dir or config.knowledge_dir
        self.processor = DocumentProcessor(self.knowledge_dir)
        self.vector_store = DenseVectorStore(embedding_dim=32, mode=mode)
        self._indexed_mode: Optional[str] = None
        self._is_indexed = False

    def initialize(self, forced_mode: Optional[str] = None) -> None:
        """Carga documentos, genera chunks y construye el índice vectorial en el modo solicitado."""
        chunks = self.processor.load_and_chunk_all()
        self.vector_store.build_index(chunks, forced_mode=forced_mode)
        self._indexed_mode = forced_mode or self.vector_store.mode
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 3, forced_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """Recupera fragmentos relevantes para una consulta."""
        if not self._is_indexed or (forced_mode and forced_mode != self._indexed_mode):
            self.initialize(forced_mode=forced_mode)
        return self.vector_store.query(query, top_k=top_k)

    def format_context_for_prompt(self, query: str, top_k: int = 3, forced_mode: Optional[str] = None) -> str:
        """Formatea los fragmentos recuperados como un bloque de contexto para inyectar en el prompt de Gemini."""
        results = self.retrieve(query, top_k=top_k, forced_mode=forced_mode)
        if not results:
            return "No se encontró información contextual específica en la base de conocimiento."

        context_blocks = []
        for i, r in enumerate(results, start=1):
            context_blocks.append(
                f"[Fragmento {i} | Fuente: {r['source_file']} | Sección: {r['section_title']} | Modo: {r.get('embedding_mode', 'N/A')} | Afinidad Semántica: {r['similarity_score']:.2f}]\n"
                f"{r['text']}"
            )
        return "\n\n".join(context_blocks)
