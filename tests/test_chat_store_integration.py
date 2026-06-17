"""
End-to-end integration tests for VastbaseChatStore.

Tests complete user flows spanning multiple method calls:
- Full CRUD lifecycle: set → get → add → delete specific → delete last → delete all
- Multiple concurrent keys
- Message persistence across multiple store instances
- Complex ChatMessage with metadata
"""

import pytest

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

from conftest import VASTBASE_CONFIG, _make_table_name


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_store():
    """Create a VastbaseChatStore for integration tests with a unique table."""
    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name=_make_table_name("integration"),
    )
    yield store
    # Cleanup: delete all keys
    try:
        keys = store.get_keys()
        for key in keys:
            store.delete_messages(key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Full CRUD lifecycle
# ---------------------------------------------------------------------------

def test_full_crud_lifecycle(integration_store: VastbaseChatStore):
    """Complete CRUD lifecycle: set → get → add → delete specific → delete
    last → delete all.

    This is the canonical integration test — if this passes, the basic
    ChatStore functionality is working.
    """
    key = "integration_crud"
    store = integration_store

    # Phase 1: SET — Create initial conversation
    store.set_messages(
        key,
        [
            ChatMessage(content="Hello", role="user"),
            ChatMessage(content="Hi there!", role="assistant"),
        ],
    )

    # Phase 2: GET — Verify initial messages
    messages = store.get_messages(key)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there!"

    # Phase 3: ADD — Append a new message
    store.add_message(
        key,
        ChatMessage(content="How are you?", role="user"),
    )
    messages = store.get_messages(key)
    assert len(messages) == 3
    assert messages[2].content == "How are you?"

    # Phase 4: DELETE_MESSAGE — Remove a specific message by index
    deleted = store.delete_message(key, 1)
    assert deleted is not None
    assert deleted.content == "Hi there!"
    messages = store.get_messages(key)
    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "How are you?"

    # Phase 5: DELETE_LAST — Remove the last message
    deleted_last = store.delete_last_message(key)
    assert deleted_last is not None
    assert deleted_last.content == "How are you?"
    messages = store.get_messages(key)
    assert len(messages) == 1
    assert messages[0].content == "Hello"

    # Phase 6: DELETE_MESSAGES — Remove all remaining messages
    store.delete_messages(key)
    assert store.get_messages(key) == []


# ---------------------------------------------------------------------------
# Multiple keys
# ---------------------------------------------------------------------------

def test_multiple_keys_independent(integration_store: VastbaseChatStore):
    """Operations on one key should not affect other keys."""
    store = integration_store

    store.set_messages(
        "user_a",
        [ChatMessage(content="A's first message", role="user")],
    )
    store.set_messages(
        "user_b",
        [ChatMessage(content="B's first message", role="user")],
    )

    # Add to user_a only
    store.add_message(
        "user_a",
        ChatMessage(content="A's second message", role="assistant"),
    )

    # User A should have 2 messages
    a_messages = store.get_messages("user_a")
    assert len(a_messages) == 2

    # User B should still have 1 message
    b_messages = store.get_messages("user_b")
    assert len(b_messages) == 1
    assert b_messages[0].content == "B's first message"

    # Delete user_b — user_a should be unaffected
    store.delete_messages("user_b")
    assert store.get_messages("user_b") == []
    assert len(store.get_messages("user_a")) == 2

    # Verify get_keys
    keys = store.get_keys()
    assert "user_a" in keys
    assert "user_b" not in keys  # Deleted


def test_keys_persistence_across_get_keys(integration_store: VastbaseChatStore):
    """Keys should persist and be consistent across get_keys calls."""
    store = integration_store

    keys_before = store.get_keys()
    assert keys_before == []

    store.set_messages(
        "session_1",
        [ChatMessage(content="S1 msg", role="user")],
    )
    store.set_messages(
        "session_2",
        [ChatMessage(content="S2 msg", role="user")],
    )

    keys_after = store.get_keys()
    assert len(keys_after) == 2
    assert "session_1" in keys_after
    assert "session_2" in keys_after


# ---------------------------------------------------------------------------
# Complex message types
# ---------------------------------------------------------------------------

def test_chat_message_additional_kwargs(
    integration_store: VastbaseChatStore,
):
    """ChatMessage with additional_kwargs should survive round-trip.

    LlamaIndex often attaches metadata to ChatMessage via additional_kwargs.
    VastbaseChatStore must preserve these through JSON serialization/deserialization.
    """
    key = "test_additional_kwargs"

    message = ChatMessage(
        role="user",
        content="Message with metadata",
        additional_kwargs={
            "tool_calls": [{"name": "search", "args": {"query": "something"}}],
            "token_count": 42,
        },
    )

    store = integration_store
    store.set_messages(key, [message])

    retrieved = store.get_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Message with metadata"
    assert retrieved[0].additional_kwargs is not None
    assert retrieved[0].additional_kwargs["tool_calls"][0]["name"] == "search"
    assert retrieved[0].additional_kwargs["token_count"] == 42


def test_message_with_all_roles(integration_store: VastbaseChatStore):
    """Test all LlamaIndex MessageRole values survive round-trip."""
    key = "test_all_roles"

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="User question"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Assistant reply"),
        ChatMessage(role=MessageRole.FUNCTION, content="Function result"),
        ChatMessage(role=MessageRole.TOOL, content="Tool output"),
    ]

    store = integration_store
    store.set_messages(key, messages)

    retrieved = store.get_messages(key)
    assert len(retrieved) == 5
    assert retrieved[0].role == MessageRole.SYSTEM
    assert retrieved[0].content == "System prompt"
    assert retrieved[1].role == MessageRole.USER
    assert retrieved[1].content == "User question"
    assert retrieved[2].role == MessageRole.ASSISTANT
    assert retrieved[3].role == MessageRole.FUNCTION
    assert retrieved[4].role == MessageRole.TOOL


# ---------------------------------------------------------------------------
# Overwrite semantics
# ---------------------------------------------------------------------------

def test_set_messages_idempotency(integration_store: VastbaseChatStore):
    """Calling set_messages multiple times with the same data should be
    idempotent."""
    key = "test_idempotent"
    messages = [
        ChatMessage(content="Same content", role="user"),
    ]

    store = integration_store
    store.set_messages(key, messages)
    store.set_messages(key, messages)
    store.set_messages(key, messages)

    retrieved = store.get_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Same content"


def test_set_messages_overwrite_shorter(integration_store: VastbaseChatStore):
    """Overwriting with fewer messages should shrink the array."""
    key = "test_shrink"

    store = integration_store
    store.set_messages(
        key,
        [
            ChatMessage(content="Msg 1", role="user"),
            ChatMessage(content="Msg 2", role="user"),
            ChatMessage(content="Msg 3", role="user"),
        ],
    )
    assert len(store.get_messages(key)) == 3

    # Overwrite with single message
    store.set_messages(
        key, [ChatMessage(content="Only one now", role="assistant")]
    )
    retrieved = store.get_messages(key)
    assert len(retrieved) == 1
    assert retrieved[0].content == "Only one now"
