from typing import List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore


class VastbaseChatStore(BaseChatStore):
    """Vastbase-backed chat store for LlamaIndex.

    Stores chat message history in Vastbase using pyvastbase Collection API,
    with one row per key and JSON-serialized message arrays.

    This is a test stub — all methods raise NotImplementedError.
    The implementation will use pyvastbase Collection API for all operations.
    """

    table_name: Optional[str] = "chatstore"
    schema_name: Optional[str] = "public"
    use_jsonb: bool = False

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "chatstore",
        schema_name: str = "public",
        use_jsonb: bool = False,
    ) -> "VastbaseChatStore":
        """Create a VastbaseChatStore from connection parameters."""
        raise NotImplementedError("VastbaseChatStore.from_params not implemented")

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        schema_name: str = "public",
        use_jsonb: bool = False,
    ) -> "VastbaseChatStore":
        """Create a VastbaseChatStore from a URI.

        URI format: vastbase://user:pass@host:port/database
        """
        raise NotImplementedError("VastbaseChatStore.from_uri not implemented")

    # ========== Synchronous methods (7) ==========

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Store messages for a key, overwriting any existing messages."""
        raise NotImplementedError("VastbaseChatStore.set_messages not implemented")

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Retrieve all messages for a given key.

        Returns empty list if key does not exist.
        """
        raise NotImplementedError("VastbaseChatStore.get_messages not implemented")

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Append a single message to the list for a key."""
        raise NotImplementedError("VastbaseChatStore.add_message not implemented")

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Delete all messages for a key. Returns None per upstream convention."""
        raise NotImplementedError("VastbaseChatStore.delete_messages not implemented")

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a specific message by index for a key.

        Returns the deleted ChatMessage, or None if key doesn't exist
        or index is out of bounds.
        """
        raise NotImplementedError("VastbaseChatStore.delete_message not implemented")

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete the last message for a key.

        Returns the deleted ChatMessage, or None if key doesn't exist
        or the message array is empty.
        """
        raise NotImplementedError("VastbaseChatStore.delete_last_message not implemented")

    def get_keys(self) -> List[str]:
        """Get all keys stored in the chat store."""
        raise NotImplementedError("VastbaseChatStore.get_keys not implemented")

    # ========== Asynchronous methods (7) ==========

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Async version of set_messages."""
        raise NotImplementedError("VastbaseChatStore.aset_messages not implemented")

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """Async version of get_messages."""
        raise NotImplementedError("VastbaseChatStore.aget_messages not implemented")

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Async version of add_message."""
        raise NotImplementedError("VastbaseChatStore.async_add_message not implemented")

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Async version of delete_messages."""
        raise NotImplementedError("VastbaseChatStore.adelete_messages not implemented")

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async version of delete_message."""
        raise NotImplementedError("VastbaseChatStore.adelete_message not implemented")

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async version of delete_last_message."""
        raise NotImplementedError("VastbaseChatStore.adelete_last_message not implemented")

    async def aget_keys(self) -> List[str]:
        """Async version of get_keys."""
        raise NotImplementedError("VastbaseChatStore.aget_keys not implemented")
