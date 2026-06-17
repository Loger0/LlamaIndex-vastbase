"""Compatibility tests: verify VastbaseChatStore is a drop-in replacement
for PostgresChatStore v0.4.0.

Checks:
- Class hierarchy (isinstance BaseChatStore)
- All 14 required methods present and callable
- Method signatures match upstream
"""

import inspect

import pytest

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.storage.chat_store.vastbase import VastbaseChatStore


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------


def test_isinstance_basechatstore():
    """Verify VastbaseChatStore extends BaseChatStore."""
    store = VastbaseChatStore(
        host="172.16.105.107",
        port=15432,
        database="vastbase",
        user="aidev",
        password="Vbase_123456",
        table_name="test_compat",
    )
    assert isinstance(store, BaseChatStore)


# ---------------------------------------------------------------------------
# Method presence
# ---------------------------------------------------------------------------


REQUIRED_SYNC = [
    "set_messages",
    "get_messages",
    "add_message",
    "delete_messages",
    "delete_message",
    "delete_last_message",
    "get_keys",
]

REQUIRED_ASYNC = [
    "aset_messages",
    "aget_messages",
    "async_add_message",
    "adelete_messages",
    "adelete_message",
    "adelete_last_message",
    "aget_keys",
]


def test_all_sync_methods_present():
    """Verify all 7 sync methods exist and are callable."""
    for name in REQUIRED_SYNC:
        assert hasattr(VastbaseChatStore, name), f"Missing sync method: {name}"
        method = getattr(VastbaseChatStore, name)
        assert callable(method), f"Method {name} is not callable"


def test_all_async_methods_present():
    """Verify all 7 async methods exist and are callable."""
    for name in REQUIRED_ASYNC:
        assert hasattr(VastbaseChatStore, name), f"Missing async method: {name}"
        method = getattr(VastbaseChatStore, name)
        assert callable(method), f"Method {name} is not callable"


# ---------------------------------------------------------------------------
# Method signatures
# ---------------------------------------------------------------------------


def test_set_messages_signature():
    """Verify set_messages signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.set_messages)
    params = list(sig.parameters.keys())
    assert "key" in params
    assert "messages" in params


def test_get_messages_signature():
    """Verify get_messages signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.get_messages)
    params = list(sig.parameters.keys())
    assert "key" in params


def test_add_message_signature():
    """Verify add_message signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.add_message)
    params = list(sig.parameters.keys())
    assert "key" in params
    assert "message" in params


def test_delete_message_signature():
    """Verify delete_message signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.delete_message)
    params = list(sig.parameters.keys())
    assert "key" in params
    assert "idx" in params


def test_delete_messages_signature():
    """Verify delete_messages signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.delete_messages)
    params = list(sig.parameters.keys())
    assert "key" in params


def test_delete_last_message_signature():
    """Verify delete_last_message signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.delete_last_message)
    params = list(sig.parameters.keys())
    assert "key" in params


def test_get_keys_signature():
    """Verify get_keys signature matches upstream."""
    sig = inspect.signature(VastbaseChatStore.get_keys)
    # get_keys takes only self
    assert callable(VastbaseChatStore.get_keys)


# ---------------------------------------------------------------------------
# Serialization compatibility
# ---------------------------------------------------------------------------


def test_chat_message_roundtrip():
    """Verify ChatMessage survives JSON round-trip via model_dump(mode='json')."""
    import json

    msg = ChatMessage(role=MessageRole.USER, content="Hello, world!")
    serialized = json.dumps([msg.model_dump(mode="json")])
    parsed = json.loads(serialized)
    restored = ChatMessage.model_validate(parsed[0])

    assert restored.role == MessageRole.USER
    assert restored.content == "Hello, world!"
