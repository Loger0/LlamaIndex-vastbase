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

### ChatStore

```python
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

# Store and retrieve messages
chat_store.set_messages("session_1", messages)
retrieved = chat_store.get_messages("session_1")
```

## Requirements

- Python >= 3.9, < 4.0
- pyvastbase >= 0.2.7
- llama-index-core >= 0.13.0, < 0.15

## License

MIT
