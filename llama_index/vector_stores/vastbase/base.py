"""VastbaseVectorStore — LlamaIndex Vastbase vector store adapter stub.

This is a RED-phase stub: the class exists so tests can be collected,
but methods raise NotImplementedError. The executor will implement them.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

import logging

_logger = logging.getLogger(__name__)


class VastbaseVectorStore(BasePydanticVectorStore):
    """Vastbase vector store for LlamaIndex.

    Faithful adapter of PGVectorStore using pyvastbase as the backend.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    # Connection params
    host: str = "localhost"
    port: int = 15432
    database: str = "vastbase"
    user: str = "aidev"
    password: str = ""

    # Collection params
    table_name: str = "llamaindex"
    schema_name: str = "public"

    # Vector config
    embed_dim: int = 1536
    use_halfvec: bool = False

    # Hybrid search config
    hybrid_search: bool = False
    text_search_config: str = "english"

    # Metadata config
    use_jsonb: bool = False
    indexed_metadata_keys: Optional[Set[Tuple[str, str]]] = None

    # Index config
    hnsw_kwargs: Optional[Dict[str, Any]] = None

    # Behavior config
    perform_setup: bool = True
    debug: bool = False
    initialization_fail_on_error: bool = False

    # Private
    _client: Any = PrivateAttr(default=None)
    _async_collection: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _collection_name: str = PrivateAttr(default=None)
    _customize_search_fn: Optional[Callable] = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = None
        self._async_collection = None
        self._is_initialized = False
        self._collection_name = f"{self.schema_name}.data_{self.table_name}"
        self._customize_search_fn = kwargs.pop("customize_search_fn", None)

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

    @property
    def client(self) -> Any:
        if not self._is_initialized:
            return None
        return self._client

    def _initialize(self) -> None:
        """Lazy initialization — stub that logs but allows test collection."""
        self._is_initialized = True

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        raise NotImplementedError("RED phase: VastbaseVectorStore.add not implemented")

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        raise NotImplementedError("RED phase: VastbaseVectorStore.async_add not implemented")

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.delete not implemented")

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.adelete not implemented")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.delete_nodes not implemented")

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.adelete_nodes not implemented")

    def clear(self) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.clear not implemented")

    async def aclear(self) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.aclear not implemented")

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError("RED phase: VastbaseVectorStore.get_nodes not implemented")

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError("RED phase: VastbaseVectorStore.aget_nodes not implemented")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError("RED phase: VastbaseVectorStore.query not implemented")

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError("RED phase: VastbaseVectorStore.aquery not implemented")

    def close(self) -> None:
        self._is_initialized = False

    def persist(
        self, persist_path: str, fs: Optional[Any] = None
    ) -> None:
        raise NotImplementedError("RED phase: VastbaseVectorStore.persist not implemented")
