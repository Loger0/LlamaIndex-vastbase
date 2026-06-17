"""
Pytest fixtures for Vastbase ChatStore tests.

Provides Vastbase connection infrastructure and VastbaseChatStore
instance fixtures for both sync and async test suites.

Note: pyvastbase imports are lazy (inside fixtures) to ensure
``pytest --collect-only`` succeeds even when pyvastbase is not
installed or has version conflicts.
"""

import os
import sys
from typing import Generator

import pytest

# Add the tests directory to sys.path so test modules can import
# ``from conftest import _make_table_name``. Do NOT add the repo
# root — it shadows the installed ``llama_index`` namespace package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NOTE: Do NOT add the repo root to sys.path — the package is installed
# in editable mode (``pip install -e .``). Adding the repo root would
# shadow the installed ``llama_index`` package and break imports of
# ``llama_index.core``.

# Vastbase connection parameters from Multica workspace environment
# Override via environment variables for CI/CD
VASTBASE_CONFIG = {
    "host": os.environ.get("VASTBASE_HOST", "172.16.105.107"),
    "port": int(os.environ.get("VASTBASE_PORT", "15432")),
    "database": os.environ.get("VASTBASE_DATABASE", "vastbase"),
    "user": os.environ.get("VASTBASE_USER", "aidev"),
    "password": os.environ.get("VASTBASE_PASSWORD", "Vbase_123456"),
}

# Unique table prefix per test run to avoid collisions
TEST_TABLE_PREFIX = os.environ.get("TEST_TABLE_PREFIX", "test_vastbase_chatstore")


def _make_table_name(base_name: str) -> str:
    """Generate a unique test table name."""
    return f"{TEST_TABLE_PREFIX}_{base_name}"


@pytest.fixture(scope="session")
def vastbase_connection():
    """Session-scoped fixture that connects to Vastbase.

    Establishes a single connection used by all tests in the session.
    Disconnected automatically at session teardown.
    """
    from pyvastbase import close_all, connect

    connect(
        host=VASTBASE_CONFIG["host"],
        port=VASTBASE_CONFIG["port"],
        database=VASTBASE_CONFIG["database"],
        user=VASTBASE_CONFIG["user"],
        password=VASTBASE_CONFIG["password"],
        alias=f"test_chatstore_{os.getpid()}",
    )
    yield VASTBASE_CONFIG
    # Cleanup: close all connections
    close_all()


@pytest.fixture
def vastbase_chat_store_params():
    """Fixture providing connection parameters for VastbaseChatStore.

    Returns a dict suitable for VastbaseChatStore.from_params().
    """
    return {
        "host": VASTBASE_CONFIG["host"],
        "port": VASTBASE_CONFIG["port"],
        "database": VASTBASE_CONFIG["database"],
        "user": VASTBASE_CONFIG["user"],
        "password": VASTBASE_CONFIG["password"],
        "table_name": _make_table_name("sync_crud"),
        "schema_name": "public",
    }
