"""
Synchronous CRUD tests for VastbaseChatStore.

Adapted from upstream PostgresChatStore test suite (llama-index-storage-chat-store-postgres v0.4.0).
All database operations use pyvastbase Collection API — no SQLAlchemy, psycopg, or raw SQL.
"""

import pytest

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

from conftest import _make_table_name


# ---------------------------------------------------------------------------
# Inheritance verification
# ---------------------------------------------------------------------------

def test_vastbase_chat_store_inherits_base_chat_store():
    """Verify VastbaseChatStore inherits from BaseChatStore.

    Source: upstream test_class() — test_chat_store_postgres_chat_store.py:25
    """
    names_of_base_classes = [b.__name__ for b in VastbaseChatStore.__mro__]
    assert BaseChatStore.__name__ in names_of_base_classes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chat_store():
    """Create a VastbaseChatStore instance for sync testing.

    Each test gets a fresh instance with a unique table name to ensure isolation.
    Cleans up all keys after each test (teardown).
    """
    store = VastbaseChatStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name=_make_table_name("sync"),
    )
    yield store
    # Teardown: delete all test data
    try:
        keys = store.get_keys()
        for key in keys:
            store.delete_messages(key)
    except Exception:
        pass  # Best-effort cleanup


# ---------------------------------------------------------------------------
# set_messages + get_messages
# ---------------------------------------------------------------------------

def test_set_and_retrieve_messages(chat_store: VastbaseChatStore):
    """Store multiple messages and retrieve them all.

    Source: upstream test_set_and_retrieve_messages() — line 105
    """
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Second message", role="user"),
    ]
    key = "test_set_key"

    chat_store.set_messages(key, messages)

    retrieved = chat_store.get_messages(key)
    assert len(retrieved) == 2
    assert retrieved[0].content == "First message"
    assert retrieved[0].role == "user"
    assert retrieved[1].content == "Second message"
    assert retrieved[1].role == "user"


def test_set_messages_overwrites_existing(chat_store: VastbaseChatStore):
    """Calling set_messages on an existing key should overwrite, not append.

    Upstream behavior: INSERT ... ON CONFLICT (key) DO UPDATE.
    VastbaseChatStore must replicate this via SELECT → upsert(id) logic.
    """
    key = "test_overwrite_key"

    # First write
    chat_store.set_messages(
        key, [ChatMessage(content="Original", role="user")]
    )
    # Overwrite with different messages
    chat_store.set_messages(
        key, [ChatMessage(content="Replaced", role="assistant")]
    )

    retrieved = chat_store.get_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Replaced"
    assert retrieved[0].role == "assistant"


def test_get_messages_nonexistent_key(chat_store: VastbaseChatStore):
    """Retrieving messages for a non-existent key should return an empty list.

    Upstream behavior: returns [] when key not found.
    """
    result = chat_store.get_messages("nonexistent_key_xyz")
    assert result == []


def test_set_messages_empty_list(chat_store: VastbaseChatStore):
    """Setting an empty message list should store an empty array.

    Upstream behavior: empty list is stored as ARRAY[] in PG.
    VastbaseChatStore should store an empty JSON array "[]".
    """
    key = "test_empty_list"

    chat_store.set_messages(key, [])

    retrieved = chat_store.get_messages(key)
    assert retrieved == []


# ---------------------------------------------------------------------------
# add_message
# ---------------------------------------------------------------------------

def test_add_message(chat_store: VastbaseChatStore):
    """Add a single message to a key and verify retrieval.

    Source: upstream test_postgres_add_message() — line 93
    """
    key = "test_add_key"

    message = ChatMessage(content="add_message_test", role="user")
    chat_store.add_message(key, message=message)

    result = chat_store.get_messages(key)
    assert len(result) == 1
    assert result[0].content == "add_message_test"
    assert result[0].role == "user"


def test_add_message_to_new_key(chat_store: VastbaseChatStore):
    """Adding a message to a key that doesn't exist yet should create it.

    Upstream uses INSERT ... ON CONFLICT DO UPDATE array_cat().
    VastbaseChatStore should: SELECT → key doesn't exist → INSERT new row.
    """
    key = "test_add_new_key"

    chat_store.add_message(
        key, ChatMessage(content="First", role="user")
    )
    chat_store.add_message(
        key, ChatMessage(content="Second", role="assistant")
    )

    result = chat_store.get_messages(key)
    assert len(result) == 2
    assert result[0].content == "First"
    assert result[1].content == "Second"


def test_add_message_appends_to_end(chat_store: VastbaseChatStore):
    """Each add_message should append to the end of the array.

    Source: upstream uses array_cat() to append.
    VastbaseChatStore should: SELECT → list.append → upsert.
    """
    key = "test_append_order"

    chat_store.set_messages(
        key, [ChatMessage(content="Base 1", role="user")]
    )
    chat_store.add_message(
        key, ChatMessage(content="Appended", role="assistant")
    )

    result = chat_store.get_messages(key)
    assert len(result) == 2
    assert result[0].content == "Base 1"
    assert result[1].content == "Appended"


# ---------------------------------------------------------------------------
# delete_messages
# ---------------------------------------------------------------------------

def test_delete_messages(chat_store: VastbaseChatStore):
    """Delete all messages for a key.

    Source: upstream test_delete_messages() — line 119
    """
    key = "test_delete_key"

    chat_store.set_messages(
        key, [ChatMessage(content="Message to delete", role="user")]
    )
    assert len(chat_store.get_messages(key)) == 1

    chat_store.delete_messages(key)
    assert chat_store.get_messages(key) == []


def test_delete_messages_nonexistent_key(chat_store: VastbaseChatStore):
    """Deleting a non-existent key should not raise an error.

    Upstream behavior: DELETE with no matching row returns silently.
    VastbaseChatStore should handle this gracefully (no-op or logged).
    """
    # Should not raise
    chat_store.delete_messages("nonexistent_key_delete")


# ---------------------------------------------------------------------------
# delete_message (by index)
# ---------------------------------------------------------------------------

def test_delete_specific_message(chat_store: VastbaseChatStore):
    """Delete a message at a specific index.

    Source: upstream test_delete_specific_message() — line 130
    """
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]
    key = "test_delete_message_key"

    chat_store.set_messages(key, messages)
    chat_store.delete_message(key, 1)

    retrieved = chat_store.get_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Keep me"


def test_delete_message_first_index(chat_store: VastbaseChatStore):
    """Delete the first message (index 0)."""
    key = "test_delete_first"
    messages = [
        ChatMessage(content="Delete me", role="user"),
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Keep me too", role="user"),
    ]

    chat_store.set_messages(key, messages)
    chat_store.delete_message(key, 0)

    retrieved = chat_store.get_messages(key)
    assert len(retrieved) == 2
    assert retrieved[0].content == "Keep me"
    assert retrieved[1].content == "Keep me too"


def test_delete_message_last_index(chat_store: VastbaseChatStore):
    """Delete the last message by explicit index."""
    key = "test_delete_last_by_idx"
    messages = [
        ChatMessage(content="Keep me", role="user"),
        ChatMessage(content="Delete me", role="user"),
    ]

    chat_store.set_messages(key, messages)
    chat_store.delete_message(key, 1)

    retrieved = chat_store.get_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Keep me"


def test_delete_message_returns_deleted(chat_store: VastbaseChatStore):
    """delete_message should return the deleted ChatMessage.

    Source: upstream delete_message returns ChatMessage.model_validate(removed).
    """
    key = "test_delete_returns"
    messages = [
        ChatMessage(content="To delete", role="assistant"),
    ]

    chat_store.set_messages(key, messages)
    deleted = chat_store.delete_message(key, 0)

    assert deleted is not None
    assert deleted.content == "To delete"
    assert deleted.role == "assistant"


def test_delete_message_nonexistent_key(chat_store: VastbaseChatStore):
    """Deleting from a non-existent key should return None.

    Upstream behavior: returns None when key doesn't exist.
    """
    deleted = chat_store.delete_message("nonexistent_key", 0)
    assert deleted is None


def test_delete_message_out_of_bounds(chat_store: VastbaseChatStore):
    """Deleting with an out-of-bounds index should return None.

    Upstream behavior: idx < 0 or idx >= len(result) → None.
    """
    key = "test_oob"
    chat_store.set_messages(
        key, [ChatMessage(content="Only message", role="user")]
    )

    # Index too large
    deleted = chat_store.delete_message(key, 99)
    assert deleted is None

    # Negative index should also return None per upstream convention
    deleted_neg = chat_store.delete_message(key, -1)
    assert deleted_neg is None

    # Existing data should remain untouched
    remaining = chat_store.get_messages(key)
    assert len(remaining) == 1


# ---------------------------------------------------------------------------
# delete_last_message
# ---------------------------------------------------------------------------

def test_delete_last_message(chat_store: VastbaseChatStore):
    """Delete the last message for a key.

    Source: upstream test_delete_last_message() — line 162
    """
    key = "test_delete_last_message"
    messages = [
        ChatMessage(content="First message", role="user"),
        ChatMessage(content="Last message", role="user"),
    ]

    chat_store.set_messages(key, messages)
    deleted = chat_store.delete_last_message(key)

    assert deleted is not None
    assert deleted.content == "Last message"
    assert deleted.role == "user"

    remaining = chat_store.get_messages(key)
    assert len(remaining) == 1
    assert remaining[0].content == "First message"


def test_delete_last_message_single_entry(chat_store: VastbaseChatStore):
    """Deleting the last message when only one exists should leave an empty list."""
    key = "test_delete_last_single"
    chat_store.set_messages(
        key, [ChatMessage(content="Only one", role="user")]
    )

    deleted = chat_store.delete_last_message(key)
    assert deleted is not None
    assert deleted.content == "Only one"

    remaining = chat_store.get_messages(key)
    assert remaining == []


def test_delete_last_message_nonexistent_key(chat_store: VastbaseChatStore):
    """Deleting last message from a non-existent key should return None.

    Upstream behavior: returns None when key doesn't exist.
    """
    deleted = chat_store.delete_last_message("nonexistent_key_last")
    assert deleted is None


def test_delete_last_message_empty_array(chat_store: VastbaseChatStore):
    """Deleting last message from a key with empty array should return None.

    Upstream behavior: len(result) == 0 → None.
    """
    key = "test_delete_last_empty"
    chat_store.set_messages(key, [])

    deleted = chat_store.delete_last_message(key)
    assert deleted is None


# ---------------------------------------------------------------------------
# get_keys
# ---------------------------------------------------------------------------

def test_get_keys(chat_store: VastbaseChatStore):
    """Get all keys from the chat store.

    Source: upstream test_get_keys() — line 146
    """
    chat_store.set_messages(
        "key1", [ChatMessage(content="Test1", role="user")]
    )
    chat_store.set_messages(
        "key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = chat_store.get_keys()
    assert "key1" in keys
    assert "key2" in keys


def test_get_keys_empty_store(chat_store: VastbaseChatStore):
    """Getting keys from an empty store should return an empty list."""
    keys = chat_store.get_keys()
    assert keys == []
