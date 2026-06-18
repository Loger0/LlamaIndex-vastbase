"""VastbaseVectorStore — LlamaIndex Vastbase vector store adapter.

Replaces PGVectorStore's SQLAlchemy + psycopg2/asyncpg + pgvector stack
with pyvastbase (VastbaseClient + Collection API).

All vector operations use pyvastbase exclusively — no raw SQL.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
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
# Awaitable helper — close() runs cleanup immediately AND supports await
# ---------------------------------------------------------------------------


async def _noop_coroutine() -> None:
    """A no-op coroutine used as the return value of ``close()``.

    ``close()`` runs all cleanup synchronously before returning this
    coroutine.  Returning a real coroutine object (from ``async def``)
    ensures compatibility with ``asyncio.run()``, which calls
    ``inspect.iscoroutine()`` — a check that custom ``__await__``-based
    awaitables like the former ``_ImmediateAwaitable`` do not pass on
    Python 3.13+.

    Calling patterns supported:

    - Sync: ``store.close()`` → cleanup runs immediately; the returned
      coroutine is discarded (never awaited).
    - Async: ``await store.close()`` → cleanup already ran; the no-op
      coroutine returns immediately.
    - ``asyncio.run(store.close())`` → works because the return value
      passes ``inspect.iscoroutine()``.
    """
    return None


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
    _async_initialized: bool = PrivateAttr(default=False)
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
        self._async_initialized = False
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

    @classmethod
    def class_name(cls) -> str:
        """Return class name for LlamaIndex component serialization."""
        return "VastbaseVectorStore"

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
            return  # Failure: do NOT mark as initialized — allow retry

        self._is_initialized = True  # Only reached on the success path

    @staticmethod
    def _patch_async_executor() -> None:
        """Patch pyvastbase AsyncExecutor.execute for named-placeholder compat.

        pyvastbase 0.2.7's ``AsyncCollection._load_schema_async()`` passes
        ``[self._name]`` (a list) to the executor, but the SQL uses
        ``%(table_name)s`` named placeholders.  psycopg 3 requires a dict
        for named placeholders, causing ``TypeError: named placeholders
        require a mapping of parameters``.

        The sync ``CollectionCore.load_schema()`` correctly passes
        ``{"table_name": self._name}`` (a dict).  This patch makes the
        async path behave consistently by converting list params to dict
        params when the SQL contains named placeholders.

        Idempotent — only patches once per process.
        """
        from pyvastbase.executor.async_impl import AsyncExecutor

        if getattr(AsyncExecutor, "_adapter_patched", False):
            return

        _original_execute = AsyncExecutor.execute

        async def _patched_execute(
            self: Any, sql: str, params: Any
        ) -> list:
            if isinstance(params, list) and params and "%(" in sql:
                import re as _re

                # Extract placeholder names in order of appearance.
                # Named placeholders can repeat (e.g. %(table_name)s
                # appearing twice); psycopg expects a dict keyed by
                # unique name, so we map each *unique* name to one
                # positional value from the list.
                all_names = _re.findall(r"%\((\w+)\)s", sql)
                seen: set = set()
                unique_names: list = []
                for n in all_names:
                    if n not in seen:
                        seen.add(n)
                        unique_names.append(n)
                if unique_names and len(unique_names) == len(params):
                    params = dict(zip(unique_names, params))
            return await _original_execute(self, sql, params)

        AsyncExecutor.execute = _patched_execute  # type: ignore[assignment]
        AsyncExecutor._adapter_patched = True  # type: ignore[attr-defined]

    async def _ensure_async_connection(self) -> None:
        """Ensure the async connection pool is established for AsyncCollection.

        AsyncCollection requires a separate async connection registered via
        ``AsyncConnections.connect()`` before it can be used.  This method
        is idempotent — once ``_async_initialized`` is True it returns
        immediately.
        """
        if self._async_initialized:
            return

        from pyvastbase import AsyncConnections

        try:
            await AsyncConnections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            # Apply monkey-patch for pyvastbase 0.2.7 async executor bug
            self._patch_async_executor()
            self._async_initialized = True
        except Exception as e:
            if self.initialization_fail_on_error:
                raise
            _logger.warning("Async connection initialization failed: %s", e)

    def _psycopg_connect(self) -> "psycopg.Connection":
        """Open a psycopg connection with autocommit for DDL operations.

        Shared helper used by ``_ensure_schema_columns`` and
        ``_ensure_auto_id_sequence`` to avoid opening multiple
        independent connections.

        Returns:
            An open ``psycopg.Connection`` with ``autocommit=True``.
            The caller is responsible for closing the connection.
        """
        import psycopg

        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
            autocommit=True,
        )

    def _build_schema_fields(self) -> list:
        """Build the list of FieldSchema for the collection.

        Returns the canonical schema definition so both the create path
        and the patch path use the same field list.
        """
        vector_dtype = (
            DataType.FLOAT16_VECTOR if self.use_halfvec else DataType.FLOAT_VECTOR
        )

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary_key=True, auto_id=True),
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

        if self.hybrid_search:
            fields.append(
                FieldSchema(name="text_search_tsv", dtype=DataType.TEXT)
            )

        return fields

    def _create_collection_if_not_exists(self) -> None:
        """Create the Vastbase collection with the required schema.

        Schema fields:
        - id         : INT64 primary key (auto-generated)
        - node_id    : VARCHAR(256) — LlamaIndex node identifier
        - ref_doc_id : VARCHAR(256) — source document identifier
        - text       : TEXT — original text content
        - metadata_  : JSON — node metadata (JSONB-compatible)
        - embedding  : FLOAT_VECTOR(embed_dim) or FLOAT16_VECTOR(embed_dim)

        When the collection already exists, missing columns are patched
        via ALTER TABLE … ADD COLUMN after checking existing columns
        through ``information_schema.columns``.  This handles stale
        collections created by older adapter versions that lacked
        columns such as ``ref_doc_id``.

        Note: pyvastbase's ``has_collection`` may fail with an API
        mismatch, and ``CREATE TABLE IF NOT EXISTS`` is a silent no-op
        when the table already exists.  Therefore, ``_ensure_schema_columns``
        is always called after create to handle the pre-existing table case.
        """
        assert self._client is not None

        fields = self._build_schema_fields()

        try:
            if self._client.has_collection(self._collection_name):
                # Collection exists — patch missing columns and ensure auto-id.
                conn = self._psycopg_connect()
                try:
                    self._ensure_schema_columns(fields, conn=conn)
                    self._ensure_auto_id_sequence(conn=conn)
                finally:
                    conn.close()
                return
        except Exception as e:
            # pyvastbase 0.2.7 has_collection has known edge cases
            # (e.g. unexpected keyword argument 'using');
            # fall through to create_collection which handles "already exists".
            _logger.debug("has_collection check failed (%s); attempting create", e)

        schema = CollectionSchema(name=self._collection_name, fields=fields)
        try:
            self._client.create_collection(self._collection_name, schema=schema)
        except Exception as e:
            err_lower = str(e).lower()
            if "already exist" not in err_lower and "duplicate" not in err_lower:
                raise
            _logger.debug(
                "Collection '%s' already exists (create raised)",
                self._collection_name,
            )

        # Always patch schema columns: CREATE TABLE IF NOT EXISTS is a
        # silent no-op for pre-existing tables with stale schemas, and
        # the has_collection path above may have been skipped due to the
        # pyvastbase API bug.  _ensure_schema_columns checks existing
        # columns via information_schema before issuing ALTER TABLE.
        #
        # Use a shared psycopg connection for both schema patching and
        # auto-id sequence setup to avoid opening two separate connections.
        try:
            conn = self._psycopg_connect()
            try:
                self._ensure_schema_columns(fields, conn=conn)
                self._ensure_auto_id_sequence(conn=conn)
            finally:
                conn.close()
        except Exception as e:
            _logger.warning(
                "Schema patch for '%s' failed: %s",
                self._collection_name,
                e,
            )

    def _ensure_auto_id_sequence(
        self, conn: Optional[Any] = None
    ) -> None:
        """Ensure the ``id`` column has a sequence-backed DEFAULT.

        pyvastbase ``FieldSchema(auto_id=True)`` correctly omits the ``id``
        column from INSERT statements, but does not generate ``SERIAL`` /
        ``GENERATED … AS IDENTITY`` DDL.  Without a DEFAULT, every INSERT
        fails with ``NotNullViolation``.

        This method creates a dedicated sequence and wires it as the column
        DEFAULT via raw SQL — idempotent across repeated calls.

        Args:
            conn: Optional open psycopg connection to reuse.  If ``None``,
                a new connection is opened via ``_psycopg_connect()`` and
                closed before returning.
        """
        assert self._client is not None

        seq_name = f"{self._collection_name}_id_seq"

        owns_conn = conn is None
        try:
            if owns_conn:
                conn = self._psycopg_connect()

            with conn.cursor() as cur:
                # Create sequence if it doesn't exist
                cur.execute(
                    "SELECT 1 FROM information_schema.sequences "
                    "WHERE sequence_name = %s AND sequence_schema = %s",
                    (seq_name, self.schema_name),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        f'CREATE SEQUENCE IF NOT EXISTS "{seq_name}" '
                        f"START WITH 1 INCREMENT BY 1"
                    )
                    _logger.debug(
                        "Created sequence '%s'", seq_name,
                    )

                # ALWAYS ensure the column DEFAULT is set — the ALTER
                # is idempotent and covers the case where the sequence
                # exists but the default was never applied.
                cur.execute(
                    "SELECT column_default "
                    "FROM information_schema.columns "
                    "WHERE table_name = %s AND table_schema = %s "
                    "AND column_name = 'id'",
                    (self._collection_name, self.schema_name),
                )
                row = cur.fetchone()
                if row is None or row[0] is None or seq_name not in str(row[0]):
                    cur.execute(
                        f'ALTER TABLE "{self._collection_name}" '
                        f"ALTER COLUMN id "
                        f"SET DEFAULT nextval('{seq_name}')"
                    )
                    _logger.debug(
                        "Set DEFAULT nextval('%s') on '%s'.id",
                        seq_name,
                        self._collection_name,
                    )
        except Exception as e:
            _logger.warning(
                "Failed to create auto-id sequence for '%s': %s",
                self._collection_name,
                e,
            )
        finally:
            if owns_conn and conn is not None:
                conn.close()

    def _ensure_schema_columns(
        self, fields: list, conn: Optional[Any] = None
    ) -> None:
        """Ensure all required columns exist in the collection table.

        Uses ``ALTER TABLE … ADD COLUMN`` via psycopg, checking existing
        columns via ``information_schema.columns`` first (Vastbase does
        not support ``ADD COLUMN IF NOT EXISTS``).  This patches stale
        collections created by older adapter versions that may be
        missing columns such as ``ref_doc_id``.

        Unlike pyvastbase's ``add_collection_field`` (which lacks the
        ``@with_executor`` decorator and fails silently), this method
        executes raw SQL directly.

        Critical columns (``node_id``, ``ref_doc_id``, ``text``,
        ``metadata_``, ``embedding``) raise ``RuntimeError`` if their
        ADD COLUMN fails — the adapter cannot function without them.
        Non-critical columns (e.g. ``text_search_tsv``) log a warning
        and are skipped.

        Args:
            fields: List of ``FieldSchema`` objects defining the target schema.
            conn: Optional open psycopg connection to reuse.  If ``None``,
                a new connection is opened via ``_psycopg_connect()`` and
                closed before returning.
        """
        # Columns the adapter cannot function without
        _CRITICAL_COLUMNS = {"node_id", "ref_doc_id", "text", "metadata_", "embedding"}

        # Map DataType enum → PostgreSQL type name
        type_map = {
            DataType.INT64: "BIGINT",
            DataType.VARCHAR: "VARCHAR({max_length})",
            DataType.TEXT: "TEXT",
            DataType.JSON: "JSONB",
            DataType.FLOAT_VECTOR: "VECTOR({dim})",
            DataType.FLOAT16_VECTOR: "HALFVECTOR({dim})",
        }

        alter_statements = []
        for field in fields:
            if field.name == "id":
                # Primary key — use BIGINT, handled separately by
                # _ensure_auto_id_sequence for the DEFAULT/sequence.
                pg_type = "BIGINT"
            else:
                pg_type = type_map.get(field.dtype)
                if pg_type is None:
                    # Unknown dtype not in type_map — skip with warning.
                    # All known dtypes are covered by type_map, so this
                    # path is only reached for future DataType additions.
                    _logger.warning(
                        "Skipping field '%s': unsupported dtype %s "
                        "(not in type_map)",
                        field.name, field.dtype,
                    )
                    continue
                if "{max_length}" in pg_type:
                    pg_type = pg_type.format(max_length=field.max_length or 256)
                if "{dim}" in pg_type:
                    pg_type = pg_type.format(dim=field.dim or self.embed_dim)

            alter_statements.append((
                field.name,
                f'ALTER TABLE "{self._collection_name}" '
                f"ADD COLUMN "
                f'"{field.name}" {pg_type}',
            ))

        if not alter_statements:
            return

        owns_conn = conn is None
        try:
            if owns_conn:
                conn = self._psycopg_connect()

            with conn.cursor() as cur:
                # Query existing columns with table_schema filter to avoid
                # false matches from other schemas with same table name.
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s AND table_schema = %s",
                    (self._collection_name, self.schema_name),
                )
                existing_cols = {row[0] for row in cur.fetchall()}

                for col_name, sql in alter_statements:
                    if col_name in existing_cols:
                        continue  # Column already exists — skip
                    try:
                        cur.execute(sql)
                    except Exception as col_err:
                        err_msg = str(col_err).lower()
                        # "already exists" / "duplicate column" → the
                        # column is present; treat as no-op regardless
                        # of whether the information_schema check
                        # caught it above.
                        if (
                            "already exist" in err_msg
                            or "duplicate" in err_msg
                        ):
                            _logger.debug(
                                "Column '%s' already exists "
                                "(detected via ADD COLUMN error)",
                                col_name,
                            )
                            continue
                        if col_name in _CRITICAL_COLUMNS:
                            raise RuntimeError(
                                f"Failed to add critical column "
                                f"'{col_name}' to "
                                f"'{self._collection_name}': "
                                f"{col_err}"
                            ) from col_err
                        _logger.warning(
                            "ADD COLUMN skipped for non-critical "
                            "column '%s': %s",
                            col_name, col_err,
                        )
            _logger.debug(
                "Schema columns ensured for '%s'", self._collection_name
            )
        finally:
            if owns_conn and conn is not None:
                conn.close()

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

    def close(self):
        """Close the VastbaseVectorStore and release all resources.

        Cleanup runs immediately on call (synchronous).  The returned
        coroutine (from ``_noop_coroutine``) allows ``await store.close()``
        and ``asyncio.run(store.close())`` to work without error.
        Closes both the sync VastbaseClient and the async collection
        if either is open.

        Calling patterns supported:

        - Sync: ``store.close()`` — cleanup runs immediately; returned
          coroutine is discarded (may emit RuntimeWarning).
        - Async: ``await store.close()`` — cleanup already ran; returns
          immediately.
        - ``asyncio.run(store.close())`` — works because return value
          passes ``inspect.iscoroutine()``.
        """
        if self._async_collection is not None:
            try:
                if hasattr(self._async_collection, "close"):
                    self._async_collection.close()
            except Exception as e:
                _logger.warning("Error closing async collection: %s", e)
            self._async_collection = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                _logger.warning("Error closing Vastbase client: %s", e)
            self._client = None
        self._is_initialized = False
        self._async_initialized = False
        return _noop_coroutine()

    async def aclose(self) -> None:
        """Alias for :meth:`close` — provided for explicit async naming.

        Delegates to ``close()`` which cleans up both the sync
        ``VastbaseClient`` and the async collection.
        """
        await self.close()

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
        source_rel = node.relationships.get(NodeRelationship.SOURCE)
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
            "metadata_": json.dumps(metadata, ensure_ascii=False),
            "embedding": embedding,
        }

    # ------------------------------------------------------------------
    # Filter translation
    # ------------------------------------------------------------------

    # Operators that require client-side filtering (PG JSONB array
    # operators ?|, ?&, @> are not reliably supported through pyvastbase
    # expr strings due to '?' conflicting with parameter placeholders).
    _CLIENT_SIDE_OPERATORS = {
        FilterOperator.ANY,
        FilterOperator.ALL,
        FilterOperator.CONTAINS,
    }

    def _build_filter_clause(
        self, filters: Optional[MetadataFilters]
    ) -> tuple:
        """Convert LlamaIndex MetadataFilters to pyvastbase expr + client filters.

        Returns:
            A tuple ``(expr_string_or_None, client_filters_or_None)``.
            *expr_string* contains only operators that can be pushed
            down to the database.  *client_filters* is a MetadataFilters
            object (or ``None``) for operators that must be evaluated in
            Python after retrieval.
        """
        if filters is None or not filters.filters:
            return None, None

        db_clauses: List[str] = []
        client_filters: List[Any] = []

        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                # Nested group — recurse
                sub_expr, sub_client = self._build_filter_clause(f)
                if sub_expr:
                    db_clauses.append(f"({sub_expr})")
                if sub_client:
                    client_filters.append(sub_client)
            else:
                op = getattr(f, "operator", None)
                if op in self._CLIENT_SIDE_OPERATORS:
                    client_filters.append(f)
                else:
                    clause = self._build_single_filter_clause(f)
                    if clause:
                        db_clauses.append(clause)

        # Build DB expression
        condition = getattr(filters, "condition", None)
        is_or = condition == FilterCondition.OR
        joiner = " OR " if is_or else " AND "

        db_expr: Optional[str] = None
        if db_clauses:
            db_expr = joiner.join(db_clauses)
            if len(db_clauses) > 1:
                db_expr = f"({db_expr})"

        # Build client-side MetadataFilters
        client_mf: Optional[MetadataFilters] = None
        if client_filters:
            client_mf = MetadataFilters(
                filters=client_filters,
                condition=condition or FilterCondition.AND,
            )

        return db_expr, client_mf

    @staticmethod
    def _escape(s: Any) -> str:
        """Escape single quotes for safe interpolation in expr strings."""
        return str(s).replace("'", "''")

    def _build_single_filter_clause(self, f: Any) -> Optional[str]:
        """Build an expr fragment for a single MetadataFilter.

        All metadata is stored in a single JSONB column ``metadata_``,
        so every filter accesses ``metadata_->>'<key>'``.
        """
        key = self._escape(f.key)
        value = f.value
        op = f.operator
        field = f"metadata_->>'{key}'"

        # --- IS_EMPTY ---
        if op == FilterOperator.IS_EMPTY:
            return f"{field} IS NULL"

        # --- IN / NIN ---
        if op in (FilterOperator.IN, FilterOperator.NIN):
            values = value if isinstance(value, (list, tuple)) else [value]
            vals_str = ", ".join(f"'{self._escape(v)}'" for v in values)
            kw = "IN" if op == FilterOperator.IN else "NOT IN"
            return f"{field} {kw} ({vals_str})"

        # --- TEXT_MATCH / TEXT_MATCH_INSENSITIVE ---
        if op in (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE):
            return f"{field} ILIKE '%{self._escape(value)}%'"

        # --- String equality / inequality ---
        if op == FilterOperator.EQ:
            return f"{field} = '{self._escape(value)}'"
        if op == FilterOperator.NE:
            return f"{field} != '{self._escape(value)}'"

        # --- Numeric comparisons (cast to float) ---
        numeric_ops = {
            FilterOperator.GT: ">",
            FilterOperator.LT: "<",
            FilterOperator.GTE: ">=",
            FilterOperator.LTE: "<=",
        }
        if op in numeric_ops:
            numeric_field = f"({field})::float"
            return f"{numeric_field} {numeric_ops[op]} {value}"

        # Fallback: treat as string equality
        _logger.warning("Unknown filter operator %s; falling back to EQ", op)
        return f"{field} = '{self._escape(value)}'"

    # ------------------------------------------------------------------
    # Client-side filter matching (ANY / ALL / CONTAINS)
    # ------------------------------------------------------------------

    @staticmethod
    def _row_matches_filter(row_meta: Dict[str, Any], f: Any) -> bool:
        """Check whether a row's metadata dict satisfies a MetadataFilter.

        Used for operators that cannot be pushed to the database
        (ANY, ALL, CONTAINS).
        """
        key = f.key
        value = f.value
        op = f.operator
        meta_val = row_meta.get(key)

        if op == FilterOperator.CONTAINS:
            if isinstance(meta_val, list):
                return value in meta_val
            if isinstance(meta_val, str):
                return value in meta_val
            return False

        if op == FilterOperator.ANY:
            if not isinstance(meta_val, list):
                return False
            check_values = value if isinstance(value, (list, tuple)) else [value]
            return any(v in meta_val for v in check_values)

        if op == FilterOperator.ALL:
            if not isinstance(meta_val, list):
                return False
            check_values = value if isinstance(value, (list, tuple)) else [value]
            return all(v in meta_val for v in check_values)

        return False

    def _apply_client_filters(
        self,
        rows: List[Dict[str, Any]],
        client_filters: Optional[MetadataFilters],
    ) -> List[Dict[str, Any]]:
        """Filter a list of row dicts in Python using client-side operators."""
        if client_filters is None:
            return rows

        result = []
        for row in rows:
            row_meta = row.get("metadata_", {}) or {}
            if isinstance(row_meta, str):
                try:
                    row_meta = json.loads(row_meta)
                except (json.JSONDecodeError, TypeError):
                    row_meta = {}
            match = self._eval_client_filters(row_meta, client_filters)
            if match:
                result.append(row)
        return result

    def _eval_client_filters(
        self, meta: Dict[str, Any], filters: MetadataFilters
    ) -> bool:
        """Evaluate nested client-side MetadataFilters against a metadata dict."""
        condition = getattr(filters, "condition", None)
        is_or = condition == FilterCondition.OR
        results: List[bool] = []

        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                results.append(self._eval_client_filters(meta, f))
            else:
                results.append(self._row_matches_filter(meta, f))

        if not results:
            return True  # no filters → match
        if is_or:
            return any(results)
        return all(results)

    # ------------------------------------------------------------------
    # Row → Node conversion helper
    # ------------------------------------------------------------------

    def _row_dict_to_node(self, row: Dict[str, Any]) -> BaseNode:
        """Convert a raw DB row dict to a LlamaIndex BaseNode.

        Handles metadata deserialization: if ``_node_content`` is present
        in the stored JSON, use ``metadata_dict_to_node`` for full fidelity;
        otherwise fall back to simple TextNode construction.
        """
        raw_metadata = row.get("metadata_", {}) or {}
        # metadata_ may come back as a JSON string from Vastbase
        if isinstance(raw_metadata, str):
            try:
                raw_metadata = json.loads(raw_metadata)
            except (json.JSONDecodeError, TypeError):
                raw_metadata = {}

        # Attempt full deserialization when _node_content is available
        if "_node_content" in raw_metadata:
            try:
                from llama_index.core.vector_stores.utils import (
                    metadata_dict_to_node,
                )
                return metadata_dict_to_node(raw_metadata, row.get("text", ""))
            except Exception:
                pass  # fall through to simple construction

        # Simple construction — strip internal keys
        node = TextNode(
            id_=row.get("node_id", ""),
            text=row.get("text", ""),
            embedding=row.get("embedding"),
        )
        node.metadata = {
            k: v
            for k, v in raw_metadata.items()
            if k not in ("_node_type", "_node_content")
        }
        return node

    # ------------------------------------------------------------------
    # CRUD — add
    # ------------------------------------------------------------------

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """Insert nodes into the Vastbase collection.

        Returns:
            The list of node IDs that were inserted.
        """
        self._initialize()
        if not nodes:
            return []

        fail_on_error = add_kwargs.get("fail_on_error", True)
        rows = [self._node_to_row_dict(n) for n in nodes]

        try:
            self._client.insert(self._collection_name, rows)
        except Exception as e:
            if fail_on_error:
                raise
            _logger.warning("add() insert failed: %s", e)
            return []

        return [n.node_id for n in nodes]

    async def async_add(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> List[str]:
        """Async version of ``add()``."""
        self._initialize()
        if not nodes:
            return []

        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        fail_on_error = kwargs.get("fail_on_error", True)
        rows = [self._node_to_row_dict(n) for n in nodes]

        try:
            col = AsyncCollection(self._collection_name)
            await col.insert(rows)
        except Exception as e:
            if fail_on_error:
                raise
            _logger.warning("async_add() insert failed: %s", e)
            return []

        return [n.node_id for n in nodes]

    # ------------------------------------------------------------------
    # CRUD — delete (by ref_doc_id)
    # ------------------------------------------------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete all rows whose ``ref_doc_id`` matches."""
        self._initialize()
        expr = f"ref_doc_id = '{self._escape(ref_doc_id)}'"
        self._client.delete(self._collection_name, expr=expr)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Async version of ``delete()``."""
        self._initialize()
        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        expr = f"ref_doc_id = '{self._escape(ref_doc_id)}'"
        col = AsyncCollection(self._collection_name)
        await col.delete(expr=expr)

    # ------------------------------------------------------------------
    # CRUD — delete_nodes
    # ------------------------------------------------------------------

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by node_ids and/or metadata filters."""
        self._initialize()
        if not node_ids and filters is None:
            return

        expr_parts: List[str] = []
        client_filters = None

        if node_ids:
            ids_str = ", ".join(f"'{self._escape(nid)}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")

        if filters is not None:
            db_expr, client_filters = self._build_filter_clause(filters)
            if db_expr:
                expr_parts.append(db_expr)

        if not expr_parts:
            return

        expr = " AND ".join(expr_parts)

        if client_filters:
            # Some filters cannot be pushed to DB — fetch then delete
            output_fields = ["node_id", "metadata_"]
            rows = self._client.query(
                self._collection_name,
                expr=expr,
                output_fields=output_fields,
            )
            matched = self._apply_client_filters(rows, client_filters)
            if matched:
                matched_ids = ", ".join(
                    f"'{self._escape(r['node_id'])}'" for r in matched
                )
                self._client.delete(
                    self._collection_name,
                    expr=f"node_id IN ({matched_ids})",
                )
        else:
            self._client.delete(self._collection_name, expr=expr)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Async version of ``delete_nodes()``."""
        self._initialize()
        if not node_ids and filters is None:
            return

        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        expr_parts: List[str] = []
        client_filters = None

        if node_ids:
            ids_str = ", ".join(f"'{self._escape(nid)}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")

        if filters is not None:
            db_expr, client_filters = self._build_filter_clause(filters)
            if db_expr:
                expr_parts.append(db_expr)

        if not expr_parts:
            return

        expr = " AND ".join(expr_parts)

        col = AsyncCollection(self._collection_name)
        if client_filters:
            rows = await col.query(
                expr=expr, output_fields=["node_id", "metadata_"]
            )
            matched = self._apply_client_filters(rows, client_filters)
            if matched:
                matched_ids = ", ".join(
                    f"'{self._escape(r['node_id'])}'" for r in matched
                )
                await col.delete(expr=f"node_id IN ({matched_ids})")
        else:
            await col.delete(expr=expr)

    # ------------------------------------------------------------------
    # CRUD — get_nodes
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[BaseNode]:
        """Retrieve nodes by node_ids and/or metadata filters."""
        self._initialize()
        if not node_ids and filters is None:
            return []

        expr_parts: List[str] = []
        client_filters = None

        if node_ids:
            ids_str = ", ".join(f"'{self._escape(nid)}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")

        if filters is not None:
            db_expr, client_filters = self._build_filter_clause(filters)
            if db_expr:
                expr_parts.append(db_expr)

        expr = " AND ".join(expr_parts) if expr_parts else None

        rows = self._client.query(
            self._collection_name,
            expr=expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )

        # Apply client-side filters
        if client_filters:
            rows = self._apply_client_filters(rows, client_filters)

        return [self._row_dict_to_node(r) for r in rows]

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[BaseNode]:
        """Async version of ``get_nodes()``."""
        self._initialize()
        if not node_ids and filters is None:
            return []

        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        expr_parts: List[str] = []
        client_filters = None

        if node_ids:
            ids_str = ", ".join(f"'{self._escape(nid)}'" for nid in node_ids)
            expr_parts.append(f"node_id IN ({ids_str})")

        if filters is not None:
            db_expr, client_filters = self._build_filter_clause(filters)
            if db_expr:
                expr_parts.append(db_expr)

        expr = " AND ".join(expr_parts) if expr_parts else None

        col = AsyncCollection(self._collection_name)
        rows = await col.query(
            expr=expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )

        if client_filters:
            rows = self._apply_client_filters(rows, client_filters)

        return [self._row_dict_to_node(r) for r in rows]

    # ------------------------------------------------------------------
    # CRUD — clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all rows from the collection."""
        self._initialize()
        self._client.truncate_collection(self._collection_name)

    async def aclear(self) -> None:
        """Async version of ``clear()``."""
        self._initialize()
        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        col = AsyncCollection(self._collection_name)
        await col.truncate()

    # ------------------------------------------------------------------
    # Query — Four-mode dispatcher (DEFAULT / SPARSE / HYBRID / MMR)
    # ------------------------------------------------------------------

    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        Dispatches to the appropriate query method based on mode:

        - **DEFAULT**: cosine similarity vector search via ``client.search()``
        - **SPARSE / TEXT_SEARCH**: ILIKE text search with client-side word-boundary scoring
        - **HYBRID**: dense vector + sparse text search, merged with deduplication
        - **MMR**: not supported — raises ``ValueError`` (matches upstream PGVectorStore)
        """
        self._initialize()

        mode = query.mode

        if mode == VectorStoreQueryMode.DEFAULT:
            return self._query_with_score(query, **kwargs)
        elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            return self._sparse_query_with_rank(query, **kwargs)
        elif mode == VectorStoreQueryMode.HYBRID:
            return self._hybrid_query(query, **kwargs)
        elif mode == VectorStoreQueryMode.MMR:
            return self._mmr_query(query, **kwargs)
        else:
            raise ValueError(f"Invalid query mode: {mode}")

    def _extract_hit_rows(self, hits: Any) -> List[Dict[str, Any]]:
        """Convert search result hits to a list of row dicts.

        Handles various hit data formats: dict, object with attributes,
        or raw dict.  Attaches ``_distance`` from each hit's distance
        attribute so scores stay aligned with rows through filtering.
        """
        rows: List[Dict[str, Any]] = []
        for hit in hits:
            row: Dict[str, Any] = {}
            if hasattr(hit, "data") and isinstance(hit.data, dict):
                row = dict(hit.data)
            elif isinstance(hit, dict):
                row = dict(hit)
            elif hasattr(hit, "data") and hit.data is not None:
                try:
                    row = {
                        k: v
                        for k, v in vars(hit.data).items()
                        if not k.startswith("_")
                    }
                except (TypeError, AttributeError):
                    pass

            if "node_id" not in row:
                row["node_id"] = str(getattr(hit, "id", ""))

            dist = getattr(hit, "distance", None)
            if dist is not None:
                row["_distance"] = float(dist)

            rows.append(row)
        return rows

    def _query_with_score(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Execute DEFAULT mode vector similarity search.

        Uses ``client.search()`` with COSINE metric type and converts
        cosine distance to similarity score (``1.0 - distance``).
        """
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for DEFAULT mode")

        db_expr, client_filters = self._build_filter_clause(query.filters)

        ef_search = kwargs.get(
            "hnsw_ef_search",
            (self.hnsw_kwargs or {}).get("hnsw_ef_search", 100),
        )

        # Build search params dict and invoke customize_search_fn if set
        search_params: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": [query.query_embedding],
            "anns_field": "embedding",
            "param": {"metric_type": "cosine", "ef": int(ef_search)},
            "limit": query.similarity_top_k,
            "expr": db_expr,
            "output_fields": ["node_id", "text", "metadata_", "embedding"],
        }

        if self._customize_search_fn is not None:
            try:
                search_params = self._customize_search_fn(search_params, **kwargs)
            except Exception as e:
                _logger.warning("customize_search_fn raised an error: %s", e)

        results = self._client.search(
            search_params["collection_name"],
            data=search_params["data"],
            anns_field=search_params["anns_field"],
            param=search_params["param"],
            limit=search_params["limit"],
            expr=search_params.get("expr"),
            output_fields=search_params["output_fields"],
        )

        hits = results[0] if results else []
        rows = self._extract_hit_rows(hits)

        if client_filters:
            rows = self._apply_client_filters(rows, client_filters)

        nodes: List[BaseNode] = []
        ids: List[str] = []
        similarities: List[float] = []

        for row in rows:
            node = self._row_dict_to_node(row)
            nodes.append(node)
            ids.append(row.get("node_id", ""))
            dist = row.pop("_distance", None)
            if dist is not None:
                similarities.append(1.0 - dist)

        return VectorStoreQueryResult(
            nodes=nodes,
            ids=ids,
            similarities=similarities if similarities else None,
        )

    # ------------------------------------------------------------------
    # Query — SPARSE / TEXT_SEARCH mode
    # ------------------------------------------------------------------

    def _sparse_query_with_rank(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Execute SPARSE / TEXT_SEARCH mode via text search.

        Uses client-side ILIKE fallback on the ``text`` field since
        Vastbase PG full-text functions (to_tsvector / to_tsquery) may
        not be available through the pyvastbase expr interface.

        Returns results ranked by keyword-match count (more matches
        → higher score).
        """
        import re

        if query.query_str is None:
            raise ValueError(
                "query_str is required for SPARSE/TEXT_SEARCH mode"
            )

        sparse_top_k = getattr(
            query, "sparse_top_k", None
        ) or query.similarity_top_k

        # Clean query string — same regex as upstream PGVectorStore
        cleaned = re.sub(r"(?!\b\.\b)\W+", " ", query.query_str).strip()
        keywords = [kw for kw in cleaned.split() if kw]

        if not keywords:
            return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

        # Build ILIKE expr on the text field for candidate retrieval.
        # Use loose substring match in SQL, then apply strict word
        # boundary scoring in Python (PG ILIKE \m/\M not available
        # through pyvastbase on all Vastbase builds).
        like_clauses = [
            f"text ILIKE '%{self._escape(kw)}%'" for kw in keywords
        ]
        text_expr = " OR ".join(like_clauses)

        # Combine with metadata filters
        db_expr, client_filters = self._build_filter_clause(query.filters)
        if db_expr:
            combined_expr = f"({text_expr}) AND ({db_expr})"
        else:
            combined_expr = f"({text_expr})"

        # Over-fetch to allow client-side word-boundary re-ranking
        fetch_limit = max(sparse_top_k * 10, 100)

        rows = self._client.query(
            self._collection_name,
            expr=combined_expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
            limit=fetch_limit,
        )

        if client_filters:
            rows = self._apply_client_filters(rows, client_filters)

        # Score by word boundary match count and sort descending

        def _word_score(text: str) -> float:
            t = text.lower()
            hits = sum(
                1
                for kw in keywords
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", t)
            )
            return hits / len(keywords) if keywords else 0.0

        scored = [(row, _word_score(row.get("text", "") or "")) for row in rows]
        scored = [(r, s) for r, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        nodes: List[BaseNode] = []
        ids: List[str] = []
        similarities: List[float] = []

        for row, score in scored[:sparse_top_k]:
            node = self._row_dict_to_node(row)
            nodes.append(node)
            ids.append(row.get("node_id", ""))
            similarities.append(score)

        return VectorStoreQueryResult(
            nodes=nodes, ids=ids, similarities=similarities
        )

    # ------------------------------------------------------------------
    # Query — HYBRID mode (dense + sparse dedup)
    # ------------------------------------------------------------------

    def _hybrid_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Execute HYBRID mode — dense vector search + sparse text search.

        Mirrors upstream PGVectorStore hybrid search behaviour:
        1. Run dense search (COSINE similarity) → top ``similarity_top_k``
        2. Run sparse search (ILIKE text match) → top ``sparse_top_k``
        3. Merge: dense results first, then unseen sparse results
        4. Deduplicate by node_id
        """
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for HYBRID mode")
        if query.query_str is None:
            raise ValueError("query_str is required for HYBRID mode")

        similarity_top_k = query.similarity_top_k
        sparse_top_k = getattr(
            query, "sparse_top_k", None
        ) or similarity_top_k

        db_expr, client_filters = self._build_filter_clause(query.filters)
        ef_search = kwargs.get(
            "hnsw_ef_search",
            (self.hnsw_kwargs or {}).get("hnsw_ef_search", 100),
        )

        # --- Dense search ---
        dense_results = self._client.search(
            self._collection_name,
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": "cosine", "ef": int(ef_search)},
            limit=similarity_top_k,
            expr=db_expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )

        dense_hits = dense_results[0] if dense_results else []
        dense_rows = self._extract_hit_rows(dense_hits)

        # --- Sparse search ---
        import re

        cleaned = re.sub(r"(?!\b\.\b)\W+", " ", query.query_str).strip()
        keywords = [kw for kw in cleaned.split() if kw]
        sparse_rows: List[Dict[str, Any]] = []

        if keywords:
            # Loose ILIKE for candidate retrieval
            like_clauses = [
                f"text ILIKE '%{self._escape(kw)}%'" for kw in keywords
            ]
            text_expr = " OR ".join(like_clauses)
            sparse_expr = f"({text_expr})"
            if db_expr:
                sparse_expr = f"{sparse_expr} AND ({db_expr})"

            # Over-fetch for client-side word-boundary filtering
            fetch_limit = max(sparse_top_k * 10, 100)
            raw_sparse = self._client.query(
                self._collection_name,
                expr=sparse_expr,
                output_fields=[
                    "node_id",
                    "text",
                    "metadata_",
                    "embedding",
                ],
                limit=fetch_limit,
            )

            # Client-side word boundary filter
            def _has_word_match(text: str) -> bool:
                t = text.lower()
                return any(
                    re.search(r"\b" + re.escape(kw.lower()) + r"\b", t)
                    for kw in keywords
                )

            sparse_rows = [
                r
                for r in raw_sparse
                if _has_word_match(r.get("text", "") or "")
            ][:sparse_top_k]

        # --- Merge: dense first, then unseen sparse ---
        seen_ids: set = set()
        merged: List[Dict[str, Any]] = []

        for row in dense_rows:
            nid = row.get("node_id", "")
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                merged.append(row)

        for row in sparse_rows:
            nid = row.get("node_id", "")
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                merged.append(row)

        # Apply client-side filters
        if client_filters:
            merged = self._apply_client_filters(merged, client_filters)

        # Build result
        nodes: List[BaseNode] = []
        ids: List[str] = []
        similarities: List[float] = []

        for row in merged:
            node = self._row_dict_to_node(row)
            nodes.append(node)
            ids.append(row.get("node_id", ""))
            dist = row.get("_distance")
            if dist is not None:
                similarities.append(1.0 - float(dist))

        return VectorStoreQueryResult(
            nodes=nodes,
            ids=ids,
            similarities=similarities if similarities else None,
        )

    # ------------------------------------------------------------------
    # Query — MMR mode (not supported — matches upstream PGVectorStore)
    # ------------------------------------------------------------------

    def _mmr_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """MMR is not supported — raises ValueError.

        Matches upstream PGVectorStore behaviour where MMR mode
        delegates to LlamaIndex's VectorIndexRetriever instead.
        """
        raise ValueError(
            "MMR is not supported in VastbaseVectorStore. "
            "Use LlamaIndex's VectorIndexRetriever for MMR reranking."
        )

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async version of ``query()``.

        Dispatches to the appropriate async query method based on mode:
        DEFAULT → ``_aquery_with_score``, SPARSE/TEXT_SEARCH →
        ``_async_sparse_query_with_rank``, HYBRID → ``_async_hybrid_query``,
        MMR → raises ValueError.
        """
        self._initialize()

        mode = query.mode

        if mode == VectorStoreQueryMode.DEFAULT:
            return await self._aquery_with_score(query, **kwargs)
        elif mode in (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH):
            return await self._async_sparse_query_with_rank(query, **kwargs)
        elif mode == VectorStoreQueryMode.HYBRID:
            return await self._async_hybrid_query(query, **kwargs)
        elif mode == VectorStoreQueryMode.MMR:
            raise ValueError(
                "MMR is not supported in VastbaseVectorStore. "
                "Use LlamaIndex's VectorIndexRetriever for MMR reranking."
            )
        else:
            raise ValueError(f"Invalid query mode: {mode}")

    async def _aquery_with_score(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async DEFAULT mode vector similarity search."""
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for DEFAULT mode")

        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        db_expr, client_filters = self._build_filter_clause(query.filters)
        ef_search = kwargs.get(
            "hnsw_ef_search",
            (self.hnsw_kwargs or {}).get("hnsw_ef_search", 100),
        )

        # Build search params dict and invoke customize_search_fn if set
        search_params: Dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": [query.query_embedding],
            "anns_field": "embedding",
            "param": {"metric_type": "cosine", "ef": int(ef_search)},
            "limit": query.similarity_top_k,
            "expr": db_expr,
            "output_fields": ["node_id", "text", "metadata_", "embedding"],
        }

        if self._customize_search_fn is not None:
            try:
                search_params = self._customize_search_fn(search_params, **kwargs)
            except Exception as e:
                _logger.warning("customize_search_fn raised an error: %s", e)

        col = AsyncCollection(search_params["collection_name"])
        results = await col.search(
            data=search_params["data"],
            anns_field=search_params["anns_field"],
            param=search_params["param"],
            limit=search_params["limit"],
            expr=search_params.get("expr"),
            output_fields=search_params["output_fields"],
        )

        hits = results[0] if results else []
        rows = self._extract_hit_rows(hits)

        if client_filters:
            rows = self._apply_client_filters(rows, client_filters)

        nodes: List[BaseNode] = []
        ids: List[str] = []
        similarities: List[float] = []

        for row in rows:
            node = self._row_dict_to_node(row)
            nodes.append(node)
            ids.append(row.get("node_id", ""))
            dist = row.pop("_distance", None)
            if dist is not None:
                similarities.append(1.0 - dist)

        return VectorStoreQueryResult(
            nodes=nodes,
            ids=ids,
            similarities=similarities if similarities else None,
        )

    async def _async_sparse_query_with_rank(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async SPARSE / TEXT_SEARCH mode via ILIKE text search."""
        import re

        if query.query_str is None:
            raise ValueError(
                "query_str is required for SPARSE/TEXT_SEARCH mode"
            )

        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        sparse_top_k = getattr(
            query, "sparse_top_k", None
        ) or query.similarity_top_k

        # Clean query string — same regex as upstream PGVectorStore
        cleaned = re.sub(r"(?!\b\.\b)\W+", " ", query.query_str).strip()
        keywords = [kw for kw in cleaned.split() if kw]

        if not keywords:
            return VectorStoreQueryResult(nodes=[], ids=[], similarities=[])

        # Build ILIKE expr — loose substring for candidate retrieval
        like_clauses = [
            f"text ILIKE '%{self._escape(kw)}%'" for kw in keywords
        ]
        text_expr = " OR ".join(like_clauses)

        db_expr, client_filters = self._build_filter_clause(query.filters)
        if db_expr:
            combined_expr = f"({text_expr}) AND ({db_expr})"
        else:
            combined_expr = f"({text_expr})"

        # Over-fetch for client-side word-boundary re-ranking
        fetch_limit = max(sparse_top_k * 10, 100)

        col = AsyncCollection(self._collection_name)
        rows = await col.query(
            expr=combined_expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
            limit=fetch_limit,
        )

        if client_filters:
            rows = self._apply_client_filters(rows, client_filters)

        # Score by word boundary match count and sort descending
        def _word_score(text: str) -> float:
            t = text.lower()
            hits = sum(
                1
                for kw in keywords
                if re.search(r"\b" + re.escape(kw.lower()) + r"\b", t)
            )
            return hits / len(keywords) if keywords else 0.0

        scored = [(row, _word_score(row.get("text", "") or "")) for row in rows]
        scored = [(r, s) for r, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        nodes: List[BaseNode] = []
        ids: List[str] = []
        similarities: List[float] = []

        for row, score in scored[:sparse_top_k]:
            node = self._row_dict_to_node(row)
            nodes.append(node)
            ids.append(row.get("node_id", ""))
            similarities.append(score)

        return VectorStoreQueryResult(
            nodes=nodes, ids=ids, similarities=similarities
        )

    async def _async_hybrid_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Async HYBRID mode — dense vector + sparse text search."""
        if query.query_embedding is None:
            raise ValueError("query_embedding is required for HYBRID mode")
        if query.query_str is None:
            raise ValueError("query_str is required for HYBRID mode")

        await self._ensure_async_connection()

        from pyvastbase import AsyncCollection

        similarity_top_k = query.similarity_top_k
        sparse_top_k = getattr(
            query, "sparse_top_k", None
        ) or similarity_top_k

        db_expr, client_filters = self._build_filter_clause(query.filters)
        ef_search = kwargs.get(
            "hnsw_ef_search",
            (self.hnsw_kwargs or {}).get("hnsw_ef_search", 100),
        )

        col = AsyncCollection(self._collection_name)

        # --- Dense search ---
        dense_results = await col.search(
            data=[query.query_embedding],
            anns_field="embedding",
            param={"metric_type": "cosine", "ef": int(ef_search)},
            limit=similarity_top_k,
            expr=db_expr,
            output_fields=["node_id", "text", "metadata_", "embedding"],
        )

        dense_hits = dense_results[0] if dense_results else []
        dense_rows = self._extract_hit_rows(dense_hits)

        # --- Sparse search ---
        import re

        cleaned = re.sub(r"(?!\b\.\b)\W+", " ", query.query_str).strip()
        keywords = [kw for kw in cleaned.split() if kw]
        sparse_rows: List[Dict[str, Any]] = []

        if keywords:
            # Loose ILIKE for candidate retrieval
            like_clauses = [
                f"text ILIKE '%{self._escape(kw)}%'" for kw in keywords
            ]
            text_expr = " OR ".join(like_clauses)
            sparse_expr = f"({text_expr})"
            if db_expr:
                sparse_expr = f"{sparse_expr} AND ({db_expr})"

            # Over-fetch for client-side word-boundary filtering
            fetch_limit = max(sparse_top_k * 10, 100)
            raw_sparse = await col.query(
                expr=sparse_expr,
                output_fields=[
                    "node_id",
                    "text",
                    "metadata_",
                    "embedding",
                ],
                limit=fetch_limit,
            )

            # Client-side word boundary filter
            def _has_word_match(text: str) -> bool:
                t = text.lower()
                return any(
                    re.search(r"\b" + re.escape(kw.lower()) + r"\b", t)
                    for kw in keywords
                )

            sparse_rows = [
                r
                for r in raw_sparse
                if _has_word_match(r.get("text", "") or "")
            ][:sparse_top_k]

        # --- Merge: dense first, then unseen sparse ---
        seen_ids: set = set()
        merged: List[Dict[str, Any]] = []

        for row in dense_rows:
            nid = row.get("node_id", "")
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                merged.append(row)

        for row in sparse_rows:
            nid = row.get("node_id", "")
            if nid and nid not in seen_ids:
                seen_ids.add(nid)
                merged.append(row)

        if client_filters:
            merged = self._apply_client_filters(merged, client_filters)

        nodes: List[BaseNode] = []
        ids: List[str] = []
        similarities: List[float] = []

        for row in merged:
            node = self._row_dict_to_node(row)
            nodes.append(node)
            ids.append(row.get("node_id", ""))
            dist = row.get("_distance")
            if dist is not None:
                similarities.append(1.0 - float(dist))

        return VectorStoreQueryResult(
            nodes=nodes,
            ids=ids,
            similarities=similarities if similarities else None,
        )

    # ------------------------------------------------------------------
    # Unsupported (matches upstream PGVectorStore behavior)
    # ------------------------------------------------------------------

    def persist(self, persist_path: str, fs: Optional[Any] = None) -> None:
        raise NotImplementedError(
            "VastbaseVectorStore does not support persist() — "
            "data is stored in Vastbase server."
        )
