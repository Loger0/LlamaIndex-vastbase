"""Stub VastbaseVectorStore — RED phase placeholder.

This stub exists ONLY so test collection succeeds.
Every method raises NotImplementedError to ensure all tests FAIL
until the real adapter is implemented.
"""

from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)


class VastbaseVectorStore(BasePydanticVectorStore):
    """Stub — replace with real implementation."""

    table_name: str = "llamaindex"
    schema_name: str = "public"
    embed_dim: int = 1536
    hybrid_search: bool = False
    text_search_config: str = "english"
    use_jsonb: bool = False
    use_halfvec: bool = False
    perform_setup: bool = True
    debug: bool = False
    initialization_fail_on_error: bool = False
    hnsw_kwargs: Optional[Dict[str, Any]] = None
    indexed_metadata_keys: Optional[Any] = None
    customize_search_fn: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _collection_name(self) -> str:
        return f"data_{self.table_name}"

    @property
    def client(self) -> Any:
        return None

    @property
    def _is_initialized(self) -> bool:
        return False

    @classmethod
    def from_params(cls, **kwargs: Any) -> "VastbaseVectorStore":
        return cls(**{k: v for k, v in kwargs.items() if k in cls.model_fields})

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        raise NotImplementedError("RED phase — not yet implemented")

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        raise NotImplementedError("RED phase — not yet implemented")

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        raise NotImplementedError("RED phase — not yet implemented")

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        raise NotImplementedError("RED phase — not yet implemented")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("RED phase — not yet implemented")

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("RED phase — not yet implemented")

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        raise NotImplementedError("RED phase — not yet implemented")

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        raise NotImplementedError("RED phase — not yet implemented")

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError("RED phase — not yet implemented")

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        raise NotImplementedError("RED phase — not yet implemented")

    def clear(self) -> None:
        raise NotImplementedError("RED phase — not yet implemented")

    async def aclear(self) -> None:
        raise NotImplementedError("RED phase — not yet implemented")

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass
