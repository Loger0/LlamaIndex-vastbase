# LlamaIndex-Vastbase

LlamaIndex storage integrations for [Vastbase](https://www.vastdata.com.cn/) — a drop-in replacement for PostgreSQL-based storage backends using the [pyvastbase](https://pypi.org/project/pyvastbase/) Collection API.

## Packages

| Package | Description |
|---------|-------------|
| `llama-index-storage-chat-store-vastbase` | Vastbase-backed chat store (drop-in replacement for `llama-index-storage-chat-store-postgres`) |

## Installation

```bash
pip install llama-index-storage-chat-store-vastbase
```

## Usage

### ChatStore — Synchronous

```python
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

# From individual parameters
chat_store = VastbaseChatStore.from_params(
    host="localhost",
    port=15432,
    database="vastbase",
    user="username",
    password="password",
    table_name="chatstore",
)

# From a URI
chat_store = VastbaseChatStore.from_uri(
    "vastbase://user:pass@localhost:15432/vastbase",
    table_name="chatstore",
)

# Store messages (overwrites existing)
chat_store.set_messages(
    key="session_1",
    messages=[
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
    ],
)

# Retrieve messages
messages = chat_store.get_messages(key="session_1")

# Append a single message
chat_store.add_message(
    key="session_1",
    message=ChatMessage(role=MessageRole.USER, content="Follow-up"),
)

# Delete the last message
deleted = chat_store.delete_last_message(key="session_1")

# Delete a specific message by index (0-based)
deleted = chat_store.delete_message(key="session_1", idx=0)

# Delete all messages for a key
all_deleted = chat_store.delete_messages(key="session_1")

# List all keys
keys = chat_store.get_keys()
```

### ChatStore — Asynchronous

All 7 sync methods have native async counterparts powered by pyvastbase `AsyncCollection`:

```python
import asyncio
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.storage.chat_store.vastbase import VastbaseChatStore


async def main():
    store = VastbaseChatStore.from_params(
        host="localhost",
        port=15432,
        database="vastbase",
        user="username",
        password="password",
        table_name="chatstore",
    )

    # Async set and get
    await store.aset_messages(
        key="async_session",
        messages=[ChatMessage(role=MessageRole.USER, content="Async hello")],
    )
    messages = await store.aget_messages(key="async_session")

    # Async add
    await store.async_add_message(
        key="async_session",
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Async reply"),
    )

    # Async delete operations
    deleted = await store.adelete_last_message(key="async_session")
    deleted = await store.adelete_message(key="async_session", idx=0)
    all_deleted = await store.adelete_messages(key="async_session")

    # Async keys
    keys = await store.aget_keys()

asyncio.run(main())
```

## API Reference

### Factory Methods

| Method | Description |
|--------|-------------|
| `from_params(host, port, database, user, password, ...)` | Create from individual connection parameters |
| `from_uri(uri, ...)` | Create from `vastbase://user:pass@host:port/db` URI |

### Synchronous Methods (7)

| Method | Description |
|--------|-------------|
| `set_messages(key, messages)` | Store messages, overwriting any existing |
| `get_messages(key)` | Retrieve all messages for a key (returns `[]` if not found) |
| `add_message(key, message)` | Append a single message (creates key if needed) |
| `delete_messages(key)` | Delete all messages for a key (returns deleted, or `None`) |
| `delete_message(key, idx)` | Delete message at 0-based index (returns deleted, or `None`) |
| `delete_last_message(key)` | Delete the last message (returns deleted, or `None`) |
| `get_keys()` | List all stored keys |

### Asynchronous Methods (7)

| Method | Description |
|--------|-------------|
| `aset_messages(key, messages)` | Async `set_messages` |
| `aget_messages(key)` | Async `get_messages` |
| `async_add_message(key, message)` | Async `add_message` |
| `adelete_messages(key)` | Async `delete_messages` |
| `adelete_message(key, idx)` | Async `delete_message` |
| `adelete_last_message(key)` | Async `delete_last_message` |
| `aget_keys()` | Async `get_keys` |

## Design

- **No SQL**: All database operations use pyvastbase `Collection` / `AsyncCollection` API — no SQLAlchemy, psycopg, asyncpg, or raw SQL.
- **Collection schema**: `{id (INT64 PK), key (VARCHAR 512), value (TEXT)}` — single row per key, messages serialized as a JSON array.
- **Array operations**: All message list manipulations (append, pop, delete-by-index) are performed in the Python layer: SELECT → list operation → upsert.
- **Drop-in compatible**: Same method signatures, return types, and error semantics as `PostgresChatStore` v0.4.0.

## Requirements

- Python >= 3.9, < 4.0
- pyvastbase >= 0.2.7
- llama-index-core >= 0.13.0, < 0.15

## License

MIT
