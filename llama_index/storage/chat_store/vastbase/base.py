"""Vastbase chat store for LlamaIndex.

Drop-in replacement for PostgresChatStore using pyvastbase Collection API.
All database operations use pyvastbase Collection/AsyncCollection — no SQLAlchemy,
psycopg, asyncpg, or raw SQL.

Reference: llama-index-storage-chat-store-postgres v0.4.0
"""

import json
import logging
import time
from typing import List, Optional
from urllib.parse import urlparse

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

logger = logging.getLogger(__name__)


class VastbaseChatStore(BaseChatStore):
    """Vastbase-backed chat store for LlamaIndex.

    Drop-in replacement for PostgresChatStore.
    Uses pyvastbase Collection API instead of SQLAlchemy + psycopg/asyncpg.

    Connection can be provided via:
        - from_params(host, port, database, user, password, ...)
        - from_uri("vastbase://user:pass@host:port/db")
        - Direct constructor with host/port/database/user/password fields

    Collection Schema:
        id (INT64, PK) | key (VARCHAR 512) | value (TEXT)

    The ``value`` column stores the entire ``List[ChatMessage]`` serialized
    as a JSON string. All array operations (append, pop, delete-by-index)
    are performed in the Python layer: SELECT → list operation → upsert.
    """

    # === Pydantic fields ===
    table_name: str = "chatstore"
    schema_name: str = "public"
    use_jsonb: bool = False  # Kept for API compat; no behavioral difference

    # Connection params
    host: str = "localhost"
    port: int = 15432
    database: str = "vastbase"
    user: str = ""
    password: str = ""

    # === Private attributes ===
    _coll: Optional[object] = PrivateAttr(default=None)
    _async_coll: Optional[object] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)
    _actual_table_name: str = PrivateAttr(default="")
    _id_counter: int = PrivateAttr(default=0)

    def __init__(self, **data):
        """Initialize VastbaseChatStore.

        Accepts all BaseChatStore + VastbaseChatStore Pydantic fields.
        Call _initialize() to set up connection and Collection.
        """
        super().__init__(**data)
        self._coll = None
        self._async_coll = None
        self._initialized = False
        self._actual_table_name = ""
        self._id_counter = 0

    # ==================================================================
    # Factory Methods
    # ==================================================================

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
        """Create VastbaseChatStore from individual connection parameters.

        Args:
            host: Vastbase server host (default: localhost).
            port: Vastbase server port (default: 15432).
            database: Database name (default: vastbase).
            user: Database user.
            password: Database password.
            table_name: Name of the chat store collection.
            schema_name: Schema name (reserved for future use).
            use_jsonb: Kept for upstream API compatibility (no-op).

        Returns:
            Initialized VastbaseChatStore instance.
        """
        store = cls(
            host=host or "localhost",
            port=port or 15432,
            database=database or "vastbase",
            user=user or "",
            password=password or "",
            table_name=table_name.lower() if table_name else "chatstore",
            schema_name=schema_name,
            use_jsonb=use_jsonb,
        )
        store._initialize()
        return store

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        schema_name: str = "public",
        use_jsonb: bool = False,
    ) -> "VastbaseChatStore":
        """Create VastbaseChatStore from a vastbase:// URI.

        URI format: vastbase://user:pass@host:port/database

        Args:
            uri: Connection URI with vastbase:// scheme.
            table_name: Name of the chat store collection.
            schema_name: Schema name (reserved for future use).
            use_jsonb: Kept for upstream API compatibility (no-op).

        Returns:
            Initialized VastbaseChatStore instance.

        Raises:
            ValueError: If the URI scheme is not 'vastbase'.
        """
        parsed = urlparse(uri)

        if parsed.scheme != "vastbase":
            raise ValueError(
                f"Unsupported URI scheme: '{parsed.scheme}'. "
                "Expected 'vastbase://'."
            )

        host = parsed.hostname or "localhost"
        port = parsed.port or 15432
        database = parsed.path.lstrip("/") if parsed.path else "vastbase"
        user = parsed.username or ""
        password = parsed.password or ""

        store = cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            table_name=table_name.lower() if table_name else "chatstore",
            schema_name=schema_name,
            use_jsonb=use_jsonb,
        )
        store._initialize()
        return store

    # ==================================================================
    # Initialization
    # ==================================================================

    def _initialize(self) -> None:
        """Set up pyvastbase connection and Collection.

        1. Connect to Vastbase (sync, using global connection pool).
        2. Check for legacy table name (``data_{table_name}``).
        3. Define CollectionSchema: id (INT64 PK), key (VARCHAR 512),
           value (TEXT).
        4. Create or open Collection.
        """
        from pyvastbase import Collection, connect, has_collection
        from pyvastbase import CollectionSchema, DataType, FieldSchema

        alias = f"chatstore_{self.table_name}"

        # 1. Establish connection (both default and named alias)
        #    Utility functions like has_collection/drop_collection use the
        #    'default' connection, so we register with both.
        conn_kwargs = dict(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        try:
            connect(**conn_kwargs, using="default")
        except Exception:
            logger.debug("Default connection already exists, reusing.")
        try:
            connect(**conn_kwargs, using=alias)
        except Exception:
            logger.debug(
                "Connection already exists for alias '%s', reusing.", alias
            )

        # 2. Legacy table detection
        legacy_name = f"data_{self.table_name}"
        if has_collection(legacy_name):
            self._actual_table_name = legacy_name
            logger.info("Using legacy collection name: %s", legacy_name)
        else:
            self._actual_table_name = self.table_name

        # 3. Define schema
        schema = CollectionSchema(
            name=self._actual_table_name,
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary_key=True,
                ),
                FieldSchema(
                    name="key",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                ),
                FieldSchema(
                    name="value",
                    dtype=DataType.TEXT,
                ),
            ],
        )

        # 4. Create or open Collection
        if has_collection(self._actual_table_name):
            self._coll = Collection(self._actual_table_name)
            logger.info(
                "Collection '%s' already exists, reusing.",
                self._actual_table_name,
            )
        else:
            self._coll = Collection(
                self._actual_table_name, schema=schema
            )
            self._coll.create()
            logger.info(
                "Collection '%s' created.", self._actual_table_name
            )

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before use (lazy init)."""
        if not self._initialized:
            self._initialize()

    def _next_id(self) -> int:
        """Generate the next unique ID for a new row.

        Uses a timestamp-based counter: ``int(time.time() * 1_000_000) + counter``.
        This is a pure Python operation with no database calls, making it safe
        to call from both synchronous and asynchronous contexts.

        The microsecond timestamp prefix provides reasonable uniqueness across
        store instances, and the counter ensures monotonicity within a single
        instance.
        """
        self._id_counter += 1
        return int(time.time() * 1_000_000) + self._id_counter

    @staticmethod
    def _escape(value: str) -> str:
        """Escape single quotes in string values for pyvastbase expr."""
        return value.replace("'", "''")

    # ==================================================================
    # Synchronous Methods — Core CRUD (Wave 1)
    # ==================================================================

    def set_messages(
        self, key: str, messages: List[ChatMessage]
    ) -> None:
        """Store messages for a key, overwriting any existing messages.

        Args:
            key: Unique identifier for the conversation.
            messages: List of ChatMessage objects to store.
        """
        self._ensure_initialized()

        # Serialize all messages to JSON
        value = json.dumps([m.model_dump(mode="json") for m in messages])

        # Check if key already exists
        existing = self._coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if existing:
            # Update existing row via upsert (on PK id)
            self._coll.upsert([
                {"id": existing[0]["id"], "key": key, "value": value}
            ])
        else:
            # Insert new row with generated id
            self._coll.insert([
                {"id": self._next_id(), "key": key, "value": value}
            ])

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Retrieve messages for a key.

        Args:
            key: Unique identifier for the conversation.

        Returns:
            List of ChatMessage objects, or empty list if key not found.
        """
        self._ensure_initialized()

        results = self._coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return []

        try:
            parsed = json.loads(results[0]["value"])
            return [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Failed to deserialize messages for key '%s'", key
            )
            return []

    def add_message(
        self, key: str, message: ChatMessage
    ) -> None:
        """Append a single message to the list for a key.

        If the key does not exist, a new entry is created.

        Args:
            key: Unique identifier for the conversation.
            message: ChatMessage to append.
        """
        self._ensure_initialized()

        existing = self._coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not existing:
            # Key doesn't exist — insert new row
            self._coll.insert([
                {
                    "id": self._next_id(),
                    "key": key,
                    "value": json.dumps([message.model_dump(mode="json")]),
                }
            ])
        else:
            # Key exists — append to array in Python, then upsert
            messages = json.loads(existing[0]["value"])
            messages.append(message.model_dump(mode="json"))
            self._coll.upsert([
                {
                    "id": existing[0]["id"],
                    "key": key,
                    "value": json.dumps(messages),
                }
            ])

    # ==================================================================
    # Synchronous Methods — Delete & Keys (Waves 2-3)
    # ==================================================================

    def delete_messages(
        self, key: str
    ) -> Optional[List[ChatMessage]]:
        """Delete all messages for a key.

        Args:
            key: Unique identifier for the conversation.

        Returns:
            List of deleted ChatMessage objects, or None if key not found.
        """
        self._ensure_initialized()

        results = self._coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        # Deserialize before deleting
        try:
            parsed = json.loads(results[0]["value"])
            messages = [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            messages = []

        # Delete the row
        self._coll.delete(expr=f"key = '{self._escape(key)}'")

        return messages

    def delete_message(
        self, key: str, idx: int
    ) -> Optional[ChatMessage]:
        """Delete a single message at the given index for a key.

        Args:
            key: Unique identifier for the conversation.
            idx: Index of the message to delete (0-based).

        Returns:
            The deleted ChatMessage, or None if key/index not found.
        """
        self._ensure_initialized()

        results = self._coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if idx < 0 or idx >= len(messages):
            return None

        removed = messages.pop(idx)

        self._coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)

    def delete_last_message(
        self, key: str
    ) -> Optional[ChatMessage]:
        """Delete the last message for a key.

        Args:
            key: Unique identifier for the conversation.

        Returns:
            The deleted ChatMessage, or None if key not found or list empty.
        """
        self._ensure_initialized()

        results = self._coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if len(messages) == 0:
            return None

        removed = messages.pop()

        self._coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)

    def get_keys(self) -> List[str]:
        """Retrieve all keys stored in the chat store.

        Returns:
            List of key strings.
        """
        self._ensure_initialized()

        results = self._coll.query(
            expr="1=1", output_fields=["key"]
        )

        return [r["key"] for r in results]

    # ==================================================================
    # Asynchronous Methods (Wave 4 — native pyvastbase AsyncCollection)
    # ==================================================================

    async def _ensure_async_initialized(self) -> None:
        """Ensure the async Collection is available (lazy init)."""
        if self._async_coll is None:
            from pyvastbase import AsyncCollection, AsyncConnections

            alias = f"chatstore_{self._actual_table_name or self.table_name}"
            await AsyncConnections.connect(
                alias,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            self._async_coll = AsyncCollection(
                self._actual_table_name or self.table_name,
                using=alias,
            )

    async def aset_messages(
        self, key: str, messages: List[ChatMessage]
    ) -> None:
        """Async version of set_messages."""
        await self._ensure_async_initialized()

        value = json.dumps([m.model_dump(mode="json") for m in messages])

        existing = await self._async_coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if existing:
            await self._async_coll.upsert([
                {"id": existing[0]["id"], "key": key, "value": value}
            ])
        else:
            await self._async_coll.insert([
                {"id": self._next_id(), "key": key, "value": value}
            ])

    async def aget_messages(
        self, key: str
    ) -> List[ChatMessage]:
        """Async version of get_messages."""
        await self._ensure_async_initialized()

        results = await self._async_coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return []

        try:
            parsed = json.loads(results[0]["value"])
            return [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Async: failed to deserialize messages for key '%s'", key
            )
            return []

    async def async_add_message(
        self, key: str, message: ChatMessage
    ) -> None:
        """Async version of add_message."""
        await self._ensure_async_initialized()

        existing = await self._async_coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not existing:
            await self._async_coll.insert([
                {
                    "id": self._next_id(),
                    "key": key,
                    "value": json.dumps([message.model_dump(mode="json")]),
                }
            ])
        else:
            messages = json.loads(existing[0]["value"])
            messages.append(message.model_dump(mode="json"))
            await self._async_coll.upsert([
                {
                    "id": existing[0]["id"],
                    "key": key,
                    "value": json.dumps(messages),
                }
            ])

    async def adelete_messages(
        self, key: str
    ) -> Optional[List[ChatMessage]]:
        """Async version of delete_messages."""
        await self._ensure_async_initialized()

        results = await self._async_coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            parsed = json.loads(results[0]["value"])
            messages = [ChatMessage.model_validate(m) for m in parsed]
        except (json.JSONDecodeError, KeyError, TypeError):
            messages = []

        await self._async_coll.delete(
            expr=f"key = '{self._escape(key)}'"
        )

        return messages

    async def adelete_message(
        self, key: str, idx: int
    ) -> Optional[ChatMessage]:
        """Async version of delete_message."""
        await self._ensure_async_initialized()

        results = await self._async_coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if idx < 0 or idx >= len(messages):
            return None

        removed = messages.pop(idx)

        await self._async_coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)

    async def adelete_last_message(
        self, key: str
    ) -> Optional[ChatMessage]:
        """Async version of delete_last_message."""
        await self._ensure_async_initialized()

        results = await self._async_coll.query(
            expr=f"key = '{self._escape(key)}'", limit=1
        )

        if not results:
            return None

        try:
            value = results[0].get("value", "[]")
            if not value or value == "[]":
                return None
            messages = json.loads(value)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        if len(messages) == 0:
            return None

        removed = messages.pop()

        await self._async_coll.upsert([
            {
                "id": results[0]["id"],
                "key": key,
                "value": json.dumps(messages),
            }
        ])

        return ChatMessage.model_validate(removed)

    async def aget_keys(self) -> List[str]:
        """Async version of get_keys."""
        await self._ensure_async_initialized()

        results = await self._async_coll.query(
            expr="1=1", output_fields=["key"]
        )

        return [r["key"] for r in results]
