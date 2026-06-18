"""VastbaseVectorStore — LlamaIndex Vastbase vector store adapter.

Replaces PGVectorStore's SQLAlchemy + psycopg2/asyncpg + pgvector stack
with pyvastbase (VastbaseClient + Collection API).

All vector operations use pyvastbase exclusively — no raw SQL.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from pyvastbase import VastbaseClient
from pyvastbase import (
    CollectionSchema,
    DataType,
    FieldSchema,
    IndexParams,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DBEmbeddingRow:
    """Represents a single row from the embedding collection.

    Mirrors the upstream PGVectorStore ``DBEmbeddingRow`` namedtuple,
    adapted for pyvastbase result dicts.
    """

    node_id: str
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    embedding: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase vector store for LlamaIndex.

    Faithful adapter of PGVectorStore using pyvastbase as the backend.
    All vector operations go through the pyvastbase Collection API.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    # ===== Connection params =====
    host: str = "localhost"
    port: int = 15432
    database: str = "vastbase"
    user: str = "aidev"
    password: str = ""

    # ===== Collection params =====
    table_name: str = "llamaindex"
    schema_name: str = "public"

    # ===== Vector config =====
    embed_dim: int = 1536
    use_halfvec: bool = False

    # ===== Hybrid search config =====
    hybrid_search: bool = False
    text_search_config: str = "english"

    # ===== Metadata config =====
    use_jsonb: bool = False
    indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None

    # ===== Index config =====
    hnsw_kwargs: Optional[Dict[str, Any]] = None

    # ===== Behavior config =====
    perform_setup: bool = True
    debug: bool = False
    initialization_fail_on_error: bool = False

    # ===== Private attributes =====
    _client: Any = PrivateAttr(default=None)
    _async_collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _collection_name: str = PrivateAttr(default=None)
    _customize_search_fn: Optional[Callable] = PrivateAttr(default=None)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:
        # Pop non-Pydantic fields BEFORE super().__init__() to prevent
        # Pydantic validation errors for unknown fields.
        customize_search_fn = kwargs.pop("customize_search_fn", None)
        super().__init__(**kwargs)
        self._client = None
        self._async_collection = None
        self._is_initialized = False
        self._collection_name = f"data_{self.table_name}"
        self._customize_search_fn = customize_search_fn

    @classmethod
    def from_params(
        cls,
        host: str = "localhost",
        port: int = 15432,
        database: str = "vastbase",
        user: str = "aidev",
        password: str = "",
        table_name: str = "llamaindex",
        schema_name: str = "public",
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        use_jsonb: bool = False,
        use_halfvec: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        perform_setup: bool = True,
        debug: bool = False,
        initialization_fail_on_error: bool = False,
        indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None,
        customize_search_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> "VastbaseVectorStore":
        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            use_jsonb=use_jsonb,
            use_halfvec=use_halfvec,
            hnsw_kwargs=hnsw_kwargs,
            perform_setup=perform_setup,
            debug=debug,
            initialization_fail_on_error=initialization_fail_on_error,
            indexed_metadata_keys=indexed_metadata_keys,
            customize_search_fn=customize_search_fn,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Return the underlying VastbaseClient, or None before initialization."""
        if not self._is_initialized:
            return None
        return self._client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Lazy initialization: create VastbaseClient + Collection + indexes.

        Called internally before any data operation.  Idempotent — once
        ``_is_initialized`` is True the method returns immediately.

        Error handling follows the upstream PGVectorStore pattern (D-03):
        - ``initialization_fail_on_error=True``  → re-raise the exception
        - ``initialization_fail_on_error=False`` → log warning, continue
        """
        if self._is_initialized:
            return

        try:
            self._client = VastbaseClient(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )

            if self.perform_setup:
                self._create_collection_if_not_exists()
                if self.hnsw_kwargs:
                    self._create_hnsw_index()
        except Exception as e:
            if self.initialization_fail_on_error:
                raise
            _logger.warning("VastbaseVectorStore initialization failed: %s", e)

        self._is_initialized = True

    def _create_collection_if_not_exists(self) -> None:
        """Create the Vastbase collection with the required schema.

        Schema fields:
        - id         : INT64 primary key (auto-generated)
        - node_id    : VARCHAR(256) — LlamaIndex node identifier
        - ref_doc_id : VARCHAR(256) — source document identifier
        - text       : TEXT — original text content
        - metadata_  : JSON — node metadata (JSONB-compatible)
        - embedding  : FLOAT_VECTOR(embed_dim) or FLOAT16_VECTOR(embed_dim)
        """
        assert self._client is not None

        try:
            if self._client.has_collection(self._collection_name):
                return
        except Exception as e:
            # pyvastbase 0.2.7 has_collection has known edge cases;
            # fall through to create_collection which handles "already exists".
            _logger.debug("has_collection check failed (%s); attempting create", e)

        vector_dtype = (
            DataType.FLOAT16_VECTOR if self.use_halfvec else DataType.FLOAT_VECTOR
        )

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(
                name="ref_doc_id", dtype=DataType.VARCHAR, max_length=256
            ),
            FieldSchema(name="text", dtype=DataType.TEXT),
            FieldSchema(name="metadata_", dtype=DataType.JSON),
            FieldSchema(
                name="embedding", dtype=vector_dtype, dim=self.embed_dim
            ),
        ]

        # When hybrid_search is enabled, add a text-search column for
        # PG-compatible full-text search (to_tsvector / to_tsquery).
        if self.hybrid_search:
            fields.append(
                FieldSchema(name="text_search_tsv", dtype=DataType.TEXT)
            )

        schema = CollectionSchema(name=self._collection_name, fields=fields)
        try:
            self._client.create_collection(self._collection_name, schema=schema)
        except Exception as e:
            err_lower = str(e).lower()
            if "already exist" not in err_lower and "duplicate" not in err_lower:
                raise
            _logger.debug(
                "Collection '%s' already exists; skipping creation",
                self._collection_name,
            )

    def _create_hnsw_index(self) -> None:
        """Create a HNSW graph index on the embedding field.

        Uses pyvastbase ``IndexParams.graph_index()`` instead of raw DDL.
        Parameters (``hnsw_m``, ``hnsw_ef_construction``) are read from
        the ``hnsw_kwargs`` dict.
        """
        assert self._client is not None

        hnsw_m = self.hnsw_kwargs.get("hnsw_m", 16)
        hnsw_ef_construction = self.hnsw_kwargs.get("hnsw_ef_construction", 64)

        index_params = IndexParams.graph_index(
            m=int(hnsw_m),
            ef_construction=int(hnsw_ef_construction),
        )
        self._client.create_index(
            self._collection_name,
            field_name="embedding",
            index_params=index_params,
        )

    def close(self) -> None:
        """Close the VastbaseClient connection and release resources.

        Synchronous — the underlying VastbaseClient uses psycopg (sync driver).
        The conftest fixtures wrap this in run_until_complete for compatibility
        with async test patterns; the try/except absorbs the TypeError if close
        is not a coroutine.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                _logger.warning("Error closing Vastbase client: %s", e)
            self._client = None
        self._async_collection = None
        self._is_initialized = False

    async def aclose(self) -> None:
        """Async close for use with AsyncCollection."""
        if self._async_collection is not None:
            try:
                if hasattr(self._async_collection, "close"):
                    await self._async_collection.close()
            except Exception as e:
                _logger.warning("Error closing async collection: %s", e)
            self._async_collection = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._is_initialized = False

    # ------------------------------------------------------------------
    # Data conversion helpers
    # ------------------------------------------------------------------

    def _node_to_row_dict(self, node: BaseNode) -> Dict[str, Any]:
        """Convert a LlamaIndex BaseNode to a dict for pyvastbase insert.

        Extracts ``ref_doc_id`` from the node's SOURCE relationship,
        mirroring upstream PGVectorStore's ``_node_to_row_dict``.
        """
        # Determine ref_doc_id from SOURCE relationship
        ref_doc_id = node.node_id
        source_rel = node.relationships.get("SOURCE")
        if source_rel is not None:
            ref_doc_id = source_rel.node_id

        # Build metadata dict (same as upstream _node_to_metadata_dict)
        metadata = node_to_metadata_dict(
            node, remove_text=True, flat_metadata=self.flat_metadata
        )

        embedding = node.get_embedding()

        return {
            "node_id": node.node_id,
            "ref_doc_id": ref_doc_id,
            "text": node.get_content(metadata_mode="none") or "",
            "metadata_": metadata,
            "embedding": embedding,
        }

    def _db_rows_to_query_result(
        self, rows: List[Dict[str, Any]]
    ) -> VectorStoreQueryResult:
        """Convert raw DB row dicts into a LlamaIndex VectorStoreQueryResult.

        Used by the query engine (Wave 2) to transform pyvastbase search
        results into the standard LlamaIndex result format.
        """
        nodes: List[BaseNode] = []
        ids: List[str] = []
        scores: List[float] = []

        for row in rows:
            node = TextNode(
                id_=row.get("node_id", ""),
                text=row.get("text", ""),
                embedding=row.get("embedding"),
            )
            # Restore metadata from the metadata_ JSON field
            raw_metadata = row.get("metadata_", {}) or {}
            if "_node_type" in raw_metadata:
                raw_metadata.pop("_node_type")
            if "_node_content" in raw_metadata:
                import json

                try:
                    node_content = json.loads(raw_metadata.pop("_node_content"))
                    # Merge node content fields into the node
                except (json.JSONDecodeError, TypeError):
                    node_content = {}
            node.metadata = {
                k: v
                for k, v in raw_metadata.items()
                if k not in ("_node_type", "_node_content")
            }

            nodes.append(node)
            ids.append(row.get("node_id", ""))

            score = row.get("score") or row.get("distance")
            if score is not None:
                scores.append(float(score))

        return VectorStoreQueryResult(
            nodes=nodes,
            ids=ids,
            similarities=scores if scores else None,
        )

    # ------------------------------------------------------------------
    # CRUD stubs — implemented in Wave 1
    # ------------------------------------------------------------------

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        raise NotImplementedError("Wave 1: VastbaseVectorStore.add not yet implemented")

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.async_add not yet implemented"
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("Wave 1: VastbaseVectorStore.delete not yet implemented")

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.adelete not yet implemented"
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.delete_nodes not yet implemented"
        )

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.adelete_nodes not yet implemented"
        )

    def clear(self) -> None:
        raise NotImplementedError("Wave 1: VastbaseVectorStore.clear not yet implemented")

    async def aclear(self) -> None:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.aclear not yet implemented"
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.get_nodes not yet implemented"
        )

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError(
            "Wave 1: VastbaseVectorStore.aget_nodes not yet implemented"
        )

    # ------------------------------------------------------------------
    # Query stubs — implemented in Wave 2
    # ------------------------------------------------------------------

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError(
            "Wave 2: VastbaseVectorStore.query not yet implemented"
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        raise NotImplementedError(
            "Wave 2: VastbaseVectorStore.aquery not yet implemented"
        )

    # ------------------------------------------------------------------
    # Unsupported (matches upstream PGVectorStore behavior)
    # ------------------------------------------------------------------

    def persist(self, persist_path: str, fs: Optional[Any] = None) -> None:
        raise NotImplementedError(
            "VastbaseVectorStore does not support persist() — "
            "data is stored in Vastbase server."
        )
