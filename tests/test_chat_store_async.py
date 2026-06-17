"""
Asynchronous CRUD tests for VastbaseChatStore.

Adapted from upstream PostgresChatStore async test suite
(llama-index-storage-chat-store-postgres v0.4.0).
All async operations use pyvastbase AsyncCollection API — no asyncpg or
SQLAlchemy async engine.

The VastbaseChatStore lazily initialises its async backend via
``_ensure_async_initialized()``, so tests use a plain sync fixture.
"""

import pytest
import pytest_asyncio

from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

from conftest import _make_table_name

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_TABLE = _make_table_name("async")


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_async():
    """Remove the async test collection before and after each test."""
    from pyvastbase import connect, drop_collection, has_collection

    conn_kwargs = dict(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
    )
    # Register both default (for has_collection/drop_collection) and
    # named (for AsyncCollection) connections.
    try:
        connect(**conn_kwargs, using="default")
    except Exception:
        pass
    try:
        connect(**conn_kwargs, using="test_chatstore_async")
    except Exception:
        pass
    if has_collection(TEST_TABLE):
        drop_collection(TEST_TABLE)
    yield
    if has_collection(TEST_TABLE):
        drop_collection(TEST_TABLE)


@pytest_asyncio.fixture
async def store() -> VastbaseChatStore:
    """Create a VastbaseChatStore instance for async testing.

    Returns a store whose async Collection is lazily initialised on first
    async method call via ``_ensure_async_initialized()``.
    """
    return VastbaseChatStore.from_params(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name=TEST_TABLE,
    )


# ---------------------------------------------------------------------------
# Async set_messages + get_messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_set_and_retrieve_messages(
    store: VastbaseChatStore,
):
    """Store and retrieve messages asynchronously.

    Source: upstream test_async_set_and_retrieve_messages() — line 194
    """
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Second async message", role="user"),
    ]
    key = "test_async_set_key"

    await store.aset_messages(key, messages)

    retrieved = await store.aget_messages(key)
    assert len(retrieved) == 2
    assert retrieved[0].content == "First async message"
    assert retrieved[1].content == "Second async message"


@pytest.mark.asyncio
async def test_async_set_messages_overwrites(
    store: VastbaseChatStore,
):
    """Async set_messages should overwrite existing data for the same key."""
    key = "test_async_overwrite"

    await store.aset_messages(
        key, [ChatMessage(content="Original", role="user")]
    )
    await store.aset_messages(
        key, [ChatMessage(content="Overwritten", role="assistant")]
    )

    retrieved = await store.aget_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Overwritten"


@pytest.mark.asyncio
async def test_async_get_messages_nonexistent_key(
    store: VastbaseChatStore,
):
    """Async get_messages for non-existent key should return empty list."""
    result = await store.aget_messages("async_nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# Async add_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_add_message(store: VastbaseChatStore):
    """Add a single message asynchronously.

    Source: upstream test_async_postgres_add_message() — line 182
    """
    key = "test_async_add_key"

    message = ChatMessage(content="async_add_message_test", role="user")
    await store.async_add_message(key, message=message)

    result = await store.aget_messages(key)
    assert len(result) == 1
    assert result[0].content == "async_add_message_test"
    assert result[0].role == "user"


@pytest.mark.asyncio
async def test_async_add_message_to_new_key(
    store: VastbaseChatStore,
):
    """Async add_message should create a new key if it doesn't exist."""
    key = "test_async_add_new"

    await store.async_add_message(
        key, ChatMessage(content="Msg 1", role="user")
    )
    await store.async_add_message(
        key, ChatMessage(content="Msg 2", role="assistant")
    )

    result = await store.aget_messages(key)
    assert len(result) == 2
    assert result[0].content == "Msg 1"
    assert result[1].content == "Msg 2"


# ---------------------------------------------------------------------------
# Async delete_messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_delete_messages(store: VastbaseChatStore):
    """Delete all messages for a key asynchronously.

    Source: upstream test_adelete_messages() — line 210
    """
    key = "test_async_delete_key"

    await store.aset_messages(
        key, [ChatMessage(content="Async message to delete", role="user")]
    )
    assert len(await store.aget_messages(key)) == 1

    await store.adelete_messages(key)
    retrieved = await store.aget_messages(key)
    assert retrieved == []


@pytest.mark.asyncio
async def test_async_delete_messages_nonexistent_key(
    store: VastbaseChatStore,
):
    """Async delete from non-existent key should not raise."""
    # Should not raise
    await store.adelete_messages("async_nonexistent_delete")


# ---------------------------------------------------------------------------
# Async delete_message (by index)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_delete_specific_message(
    store: VastbaseChatStore,
):
    """Delete a specific message by index asynchronously.

    Source: upstream test_async_delete_specific_message() — line 222
    """
    messages = [
        ChatMessage(content="Async keep me", role="user"),
        ChatMessage(content="Async delete me", role="user"),
    ]
    key = "test_async_delete_message_key"

    await store.aset_messages(key, messages)
    await store.adelete_message(key, 1)

    retrieved = await store.aget_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Async keep me"


@pytest.mark.asyncio
async def test_async_delete_message_returns_deleted(
    store: VastbaseChatStore,
):
    """Async delete_message should return the deleted ChatMessage."""
    key = "test_async_delete_returns"

    await store.aset_messages(
        key, [ChatMessage(content="To delete async", role="assistant")]
    )
    deleted = await store.adelete_message(key, 0)

    assert deleted is not None
    assert deleted.content == "To delete async"
    assert deleted.role == "assistant"


@pytest.mark.asyncio
async def test_async_delete_message_nonexistent_key(
    store: VastbaseChatStore,
):
    """Async delete_message on non-existent key should return None."""
    deleted = await store.adelete_message("async_nonexistent", 0)
    assert deleted is None


@pytest.mark.asyncio
async def test_async_delete_message_out_of_bounds(
    store: VastbaseChatStore,
):
    """Async delete_message with out-of-bounds index should return None."""
    key = "test_async_oob"

    await store.aset_messages(
        key, [ChatMessage(content="Only one", role="user")]
    )

    deleted = await store.adelete_message(key, 99)
    assert deleted is None

    deleted_neg = await store.adelete_message(key, -1)
    assert deleted_neg is None

    # Data should remain intact
    remaining = await store.aget_messages(key)
    assert len(remaining) == 1


# ---------------------------------------------------------------------------
# Async delete_last_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_delete_last_message(
    store: VastbaseChatStore,
):
    """Delete the last message asynchronously.

    Source: upstream test_async_delete_last_message() — line 254
    """
    key = "test_async_delete_last_message"
    messages = [
        ChatMessage(content="First async message", role="user"),
        ChatMessage(content="Last async message", role="user"),
    ]

    await store.aset_messages(key, messages)
    deleted = await store.adelete_last_message(key)

    assert deleted is not None
    assert deleted.content == "Last async message"

    remaining = await store.aget_messages(key)
    assert len(remaining) == 1
    assert remaining[0].content == "First async message"


@pytest.mark.asyncio
async def test_async_delete_last_message_nonexistent_key(
    store: VastbaseChatStore,
):
    """Async delete_last_message on non-existent key should return None."""
    deleted = await store.adelete_last_message(
        "async_nonexistent_last"
    )
    assert deleted is None


@pytest.mark.asyncio
async def test_async_delete_last_message_empty_array(
    store: VastbaseChatStore,
):
    """Async delete_last_message on empty array should return None."""
    key = "test_async_empty_last"

    await store.aset_messages(key, [])
    deleted = await store.adelete_last_message(key)
    assert deleted is None


# ---------------------------------------------------------------------------
# Async get_keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_get_keys(store: VastbaseChatStore):
    """Get all keys asynchronously.

    Source: upstream test_async_get_keys() — line 238
    """
    await store.aset_messages(
        "async_key1", [ChatMessage(content="Test1", role="user")]
    )
    await store.aset_messages(
        "async_key2", [ChatMessage(content="Test2", role="user")]
    )

    keys = await store.aget_keys()
    assert "async_key1" in keys
    assert "async_key2" in keys


@pytest.mark.asyncio
async def test_async_get_keys_empty_store(
    store: VastbaseChatStore,
):
    """Async get_keys on an empty store should return empty list."""
    keys = await store.aget_keys()
    assert keys == []


# ---------------------------------------------------------------------------
# Async multimodal messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_multimodal_messages(
    store: VastbaseChatStore,
):
    """Test that multimodal ChatMessage (TextBlock + ImageBlock) survives
    async round-trip through VastbaseChatStore.

    Source: upstream test_async_multimodal_messages() — line 274
    """
    key = "test_async_multimodal"
    image_url = "https://images.unsplash.com/photo-1579546929518-9e396f3cc809"

    messages = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="describe the image."),
                ImageBlock(url=image_url),
            ],
        )
    ]

    await store.aset_messages(key, messages)

    retrieved = await store.aget_messages(key)

    assert len(retrieved) == 1
    assert retrieved[0].role == "user"
    assert len(retrieved[0].blocks) == 2
    assert isinstance(retrieved[0].blocks[0], TextBlock)
    assert retrieved[0].blocks[0].text == "describe the image."
    assert isinstance(retrieved[0].blocks[1], ImageBlock)
    assert str(retrieved[0].blocks[1].url) == image_url
