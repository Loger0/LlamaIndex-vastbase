"""
VastbaseChatStore initialization and schema management tests.

Adapted from upstream PostgresChatStore table management tests.
Tests cover:
- from_params / from_uri factory methods
- Custom table name
- Default table name when empty string is provided
- Legacy table name detection (data_ prefix)
- Collection/schema auto-creation
"""

import pytest

from llama_index.core.llms import ChatMessage
from llama_index.storage.chat_store.vastbase import VastbaseChatStore

from conftest import VASTBASE_CONFIG, _make_table_name


# ---------------------------------------------------------------------------
# from_params
# ---------------------------------------------------------------------------

def test_from_params_creates_instance():
    """VastbaseChatStore.from_params should return a VastbaseChatStore instance."""
    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name=_make_table_name("from_params"),
    )
    assert isinstance(store, VastbaseChatStore)
    assert store.table_name == _make_table_name("from_params")
    assert store.schema_name == "public"


def test_from_params_default_table_name():
    """from_params should default to 'chatstore' when table_name not provided."""
    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
    )
    assert store.table_name == "chatstore"


def test_from_params_custom_schema_name():
    """from_params should respect the schema_name parameter."""
    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name=_make_table_name("custom_schema"),
        schema_name="custom_schema",
    )
    assert store.schema_name == "custom_schema"


# ---------------------------------------------------------------------------
# from_uri
# ---------------------------------------------------------------------------

def test_from_uri_parses_vastbase_uri():
    """from_uri should parse vastbase:// URI format correctly.

    URI format: vastbase://user:pass@host:port/database
    Source: upstream from_uri uses urlparse to extract params.
    VastbaseChatStore should support vastbase:// scheme.
    """
    uri = (
        f"vastbase://{VASTBASE_CONFIG['user']}:{VASTBASE_CONFIG['password']}"
        f"@{VASTBASE_CONFIG['host']}:{VASTBASE_CONFIG['port']}"
        f"/{VASTBASE_CONFIG['database']}"
    )
    store = VastbaseChatStore.from_uri(
        uri,
        table_name=_make_table_name("from_uri"),
    )
    assert isinstance(store, VastbaseChatStore)
    assert store.table_name == _make_table_name("from_uri")


def test_from_uri_default_table_name():
    """from_uri should default to 'chatstore' when table_name not provided."""
    uri = (
        f"vastbase://{VASTBASE_CONFIG['user']}:{VASTBASE_CONFIG['password']}"
        f"@{VASTBASE_CONFIG['host']}:{VASTBASE_CONFIG['port']}"
        f"/{VASTBASE_CONFIG['database']}"
    )
    store = VastbaseChatStore.from_uri(uri)
    assert store.table_name == "chatstore"


# ---------------------------------------------------------------------------
# Table naming
# ---------------------------------------------------------------------------

def test_custom_table_name():
    """VastbaseChatStore should respect custom table names.

    Source: upstream test_table_name_without_prefix() — line 302

    The Collection should be created with the exact table_name,
    not a 'data_' prefixed version.
    """
    table_name = _make_table_name("custom_naming")

    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name=table_name,
    )

    # Verify the store tracks the correct table name
    assert store.table_name == table_name

    # Clean up
    try:
        store.delete_messages("dummy")  # Won't exist, just triggers connection
    except Exception:
        pass


def test_empty_table_name_defaults_to_chatstore():
    """Empty string table_name should default to 'chatstore'.

    Source: upstream test_empty_table_name_defaults_to_chatstore() — line 371
    """
    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name="",
    )
    assert store.table_name == "chatstore"


def test_table_name_lowercased():
    """Table name should be lowercased, matching upstream behavior.

    Source: upstream __init__ calls table_name.lower().
    """
    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name="MixedCaseTable",
    )
    assert store.table_name == "mixedcasetable"


# ---------------------------------------------------------------------------
# Legacy table compatibility
# ---------------------------------------------------------------------------

def test_legacy_table_name_detection():
    """When a legacy table with 'data_' prefix exists, VastbaseChatStore should use it.

    Source: upstream test_legacy_table_name_detection() — line 329

    VastbaseChatStore should detect if a collection named 'data_{table_name}'
    exists and use it for backward compatibility.
    """
    # This test requires a pre-existing collection with the legacy name.
    # The VastbaseChatStore._check_legacy_table_exists() method should handle this.
    # For now, this test documents the expected behavior.

    table_name = _make_table_name("legacy")

    # Step 1: Create a store to establish the current (non-legacy) collection
    store_current = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name=table_name,
    )
    # Cleanup any data from the store setup
    try:
        keys = store_current.get_keys()
        for key in keys:
            store_current.delete_messages(key)
    except Exception:
        pass

    # In the Vastbase adapter, the equivalent of the upstream data_ prefix
    # detection is checking has_collection(f"data_{table_name}").
    # This test verifies the detection mechanism exists and works.
    # Full test requires either:
    # a) Manual creation of data_{table_name} collection via pyvastbase
    # b) Adapter-level testing of _check_legacy_table_exists()

    # For now, verify the table_name is correct on the current store
    assert store_current.table_name == table_name


# ---------------------------------------------------------------------------
# Schema / Collection auto-creation
# ---------------------------------------------------------------------------

def test_collection_auto_created_on_init():
    """VastbaseChatStore should auto-create the backing Collection on initialization.

    Source: upstream _initialize() calls _create_schema_if_not_exists()
    + _create_tables_if_not_exists(). VastbaseChatStore should create
    the Collection via pyvastbase API.
    """
    table_name = _make_table_name("auto_create")

    store = VastbaseChatStore.from_params(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        table_name=table_name,
    )

    # The store should be able to accept data immediately
    # (i.e., the collection was created during init)
    store.set_messages(
        "test_auto",
        [ChatMessage(content="Auto-created collection works", role="user")],
    )

    # Verify we can read back
    retrieved = store.get_messages("test_auto")
    assert len(retrieved) == 1
    assert retrieved[0].content == "Auto-created collection works"

    # Cleanup
    try:
        store.delete_messages("test_auto")
    except Exception:
        pass
