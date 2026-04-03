"""
Integration tests for the Tensorlake sandbox provider.

Run with:
    TENSORLAKE_API_KEY=<your-key> pytest tests/integration_test_tensorlake_sandbox.py -v -m tensorlake_sandbox

The tests follow the same structure as integration_test_async_tool_sandbox.py so
behaviour is directly comparable with the E2B and local sandbox paths.
"""

import asyncio
import os
import secrets
import string
import uuid

# Must be set before any letta imports so that settings.database_engine correctly
# detects POSTGRES. The settings singleton is created on first import of letta.settings;
# if LETTA_PG_URI is absent at that point, database_engine returns SQLITE and the
# SQLite-only INSERT OR IGNORE listener fires against the PostgreSQL connection.
os.environ.setdefault("LETTA_PG_URI", "postgresql+pg8000://letta:letta@localhost:5432/letta")

import pytest
from sqlalchemy import delete

from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.orm.sandbox_config import SandboxConfig, SandboxEnvironmentVariable
from letta.schemas.agent import AgentState
from letta.schemas.environment_variables import AgentEnvironmentVariable, SandboxEnvironmentVariableCreate
from letta.schemas.enums import ToolType
from letta.schemas.organization import Organization
from letta.schemas.pip_requirement import PipRequirement
from letta.schemas.sandbox_config import SandboxConfigCreate, TensorlakeSandboxConfig
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.services.organization_manager import OrganizationManager
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_manager import ToolManager
from letta.services.tool_sandbox.tensorlake_sandbox import AsyncToolSandboxTensorlake
from letta.services.user_manager import UserManager


def create_tool_from_func(func: callable) -> Tool:
    """Minimal inline version — avoids importing tests.helpers.utils which pulls in FastAPI routers."""
    return Tool(
        name=func.__name__,
        description="",
        source_type="python",
        tags=[],
        source_code=parse_source_code(func),
        json_schema=generate_schema(func, None),
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

namespace = uuid.NAMESPACE_DNS
org_name = str(uuid.uuid5(namespace, "test-tensorlake-sandbox-org"))
user_name = str(uuid.uuid5(namespace, "test-tensorlake-sandbox-user"))

os.environ["LETTA_DISABLE_SQLALCHEMY_POOLING"] = "true"

# ---------------------------------------------------------------------------
# Session-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_organization():
    org = await OrganizationManager().create_organization_async(Organization(name=org_name))
    yield org


@pytest.fixture
async def test_user(test_organization):
    user = await UserManager().create_actor_async(User(name=user_name, organization_id=test_organization.id))
    yield user


@pytest.fixture(autouse=True)
async def clear_tables():
    """Clear sandbox state before each test and terminate any live Tensorlake sandboxes after."""
    from letta.server.db import db_registry

    async with db_registry.async_session() as session:
        await session.execute(delete(SandboxEnvironmentVariable))
        await session.execute(delete(SandboxConfig))

    yield

    # Terminate all agent-scoped Tensorlake sandboxes kept alive during the test.
    # The production code intentionally leaves these running (Tensorlake auto-suspends
    # them), but in tests we want a clean slate after each run.
    from tensorlake.sandbox import SandboxClient, SandboxNotFoundError

    from letta.services.tool_sandbox.tensorlake_sandbox import AsyncToolSandboxTensorlake
    from letta.settings import tool_settings

    tracked = dict(AsyncToolSandboxTensorlake._agent_sandbox_ids)
    AsyncToolSandboxTensorlake._agent_sandbox_ids.clear()

    if tracked and tool_settings.tensorlake_api_key:
        import logging
        client = SandboxClient(api_key=tool_settings.tensorlake_api_key)
        for agent_id, (sandbox_id, _) in tracked.items():
            try:
                await asyncio.to_thread(client.delete, sandbox_id)
            except SandboxNotFoundError:
                pass  # already gone
            except Exception as e:
                # Non-fatal: log and continue so one failure doesn't block cleanup of others
                logging.getLogger(__name__).warning(
                    f"Failed to delete Tensorlake sandbox {sandbox_id} (agent {agent_id}): {e}"
                )


# ---------------------------------------------------------------------------
# Tool fixtures (mirrors the shared fixtures in integration_test_async_tool_sandbox.py)
# ---------------------------------------------------------------------------


@pytest.fixture
async def add_integers_tool(test_user):
    def add(x: int, y: int) -> int:
        """
        Simple function that adds two integers.

        Parameters:
            x (int): The first integer to add.
            y (int): The second integer to add.

        Returns:
            int: The sum of x and y.
        """
        return x + y

    tool = create_tool_from_func(add)
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool


@pytest.fixture
async def get_env_tool(test_user):
    def get_env() -> str:
        """
        Returns the value of the secret_word environment variable.

        Returns:
            str: The secret word.
        """
        import os

        secret_word = os.getenv("secret_word")
        print(secret_word)
        return secret_word

    tool = create_tool_from_func(get_env)
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool


@pytest.fixture
async def list_tool(test_user):
    def create_list() -> list:
        """Returns a fixed list of integers."""
        return [1, 2, 3, 4, 5]

    tool = create_tool_from_func(create_list)
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool


@pytest.fixture
async def always_err_tool(test_user):
    def error() -> str:
        """Raises an intentional error."""
        raise ZeroDivisionError("This is an intentionally weird division!")

    tool = create_tool_from_func(error)
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool


@pytest.fixture
async def write_and_read_tool(test_user):
    """Tool that writes a file and reads it back — tests persistent filesystem."""

    def write_and_read(filename: str, content: str) -> str:
        """
        Writes content to a file and reads it back.

        Parameters:
            filename (str): The name of the file to write.
            content (str): The content to write.

        Returns:
            str: The content read back from the file.
        """
        import os

        path = f"/tmp/{filename}"
        with open(path, "w") as f:
            f.write(content)
        with open(path) as f:
            return f.read()

    tool = create_tool_from_func(write_and_read)
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool


@pytest.fixture
async def agent_state():
    """
    Minimal agent-state stub for sandbox tests.
    The sandbox only reads agent_state.id and agent_state.get_agent_env_vars_as_dict();
    no LLM provider configuration is required.
    """
    import uuid
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class _StubAgentState:
        id: str
        secrets: list = dc_field(default_factory=list)

        def get_agent_env_vars_as_dict(self):
            return {s.key: s.value for s in self.secrets}

    yield _StubAgentState(id=f"agent-{uuid.uuid4()}")


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_basic_addition(check_tensorlake_key_is_set, add_integers_tool, test_user):
    """Tool runs and returns correct result."""
    args = {"x": 10, "y": 5}
    sandbox = AsyncToolSandboxTensorlake(add_integers_tool.name, args, user=test_user, tool_id=add_integers_tool.id)
    result = await sandbox.run()
    assert int(result.func_return) == 15
    assert result.status == "success"


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_list_return(check_tensorlake_key_is_set, list_tool, test_user):
    """Tool returning a list comes back as a list."""
    sandbox = AsyncToolSandboxTensorlake(list_tool.name, {}, user=test_user, tool_id=list_tool.id)
    result = await sandbox.run()
    assert result.func_return == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_error_handling(check_tensorlake_key_is_set, always_err_tool, test_user):
    """Errors inside the tool are surfaced as status=error, not raised as exceptions."""
    sandbox = AsyncToolSandboxTensorlake(always_err_tool.name, {}, user=test_user, tool_id=always_err_tool.id)
    result = await sandbox.run()
    assert result.status == "error"
    assert result.func_return is not None  # Should contain an error description


# ---------------------------------------------------------------------------
# Environment variable tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_env_var_injection(check_tensorlake_key_is_set, get_env_tool, test_user):
    """Environment variables stored in the sandbox config are injected into the tool."""
    manager = SandboxConfigManager()
    config_create = SandboxConfigCreate(config=TensorlakeSandboxConfig().model_dump())
    config = await manager.create_or_update_sandbox_config_async(config_create, test_user)

    key = "secret_word"
    expected_value = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    await manager.create_sandbox_env_var_async(
        SandboxEnvironmentVariableCreate(key=key, value=expected_value),
        sandbox_config_id=config.id,
        actor=test_user,
    )

    sandbox = AsyncToolSandboxTensorlake(get_env_tool.name, {}, user=test_user, tool_id=get_env_tool.id)
    result = await sandbox.run()
    assert expected_value in result.func_return


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_per_agent_env_overrides_global(
    check_tensorlake_key_is_set, get_env_tool, agent_state, test_user
):
    """Agent-level env vars take priority over global sandbox env vars."""
    manager = SandboxConfigManager()
    key = "secret_word"
    global_value = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))
    agent_value = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(20))

    config_create = SandboxConfigCreate(config=TensorlakeSandboxConfig().model_dump())
    config = await manager.create_or_update_sandbox_config_async(config_create, test_user)
    await manager.create_sandbox_env_var_async(
        SandboxEnvironmentVariableCreate(key=key, value=global_value),
        sandbox_config_id=config.id,
        actor=test_user,
    )

    agent_state.secrets = [AgentEnvironmentVariable(key=key, value=agent_value, agent_id=agent_state.id)]

    sandbox = AsyncToolSandboxTensorlake(get_env_tool.name, {}, user=test_user, tool_id=get_env_tool.id)
    result = await sandbox.run(agent_state=agent_state)
    assert global_value not in result.func_return
    assert agent_value in result.func_return


# ---------------------------------------------------------------------------
# Agent-persistence test (the key Tensorlake differentiator)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_persistent_filesystem_across_calls(
    check_tensorlake_key_is_set, write_and_read_tool, test_user, agent_state
):
    """
    The core Tensorlake advantage: state written in one tool call is still there
    on the next call because the sandbox is keyed to the agent and persists.

    Call 1 writes a file. Call 2 reads it back without writing again.
    If the sandbox were ephemeral (like E2B), call 2 would fail.
    """
    agent_id = agent_state.id
    filename = f"persistence_test_{uuid.uuid4().hex[:8]}.txt"
    content = "tensorlake_persistent_value_" + secrets.token_hex(8)

    # Call 1: write the file
    sandbox1 = AsyncToolSandboxTensorlake(
        write_and_read_tool.name,
        {"filename": filename, "content": content},
        user=test_user,
        tool_id=write_and_read_tool.id,
        agent_id=agent_id,
    )
    result1 = await sandbox1.run()
    assert result1.status == "success"
    assert content in result1.func_return

    # Call 2: read the same file without writing — only works if the sandbox persisted
    def read_file(filename: str) -> str:
        """
        Reads a file that was written in a previous call.

        Parameters:
            filename (str): The filename to read.

        Returns:
            str: The file contents, or an error message if the file is missing.
        """
        try:
            with open(f"/tmp/{filename}") as f:
                return f.read()
        except FileNotFoundError:
            return "FILE_NOT_FOUND"

    read_tool = create_tool_from_func(read_file)
    read_tool = await ToolManager().create_or_update_tool_async(read_tool, test_user)

    sandbox2 = AsyncToolSandboxTensorlake(
        read_tool.name,
        {"filename": filename},
        user=test_user,
        tool_id=read_tool.id,
        agent_id=agent_id,  # same agent → same sandbox
    )
    result2 = await sandbox2.run()

    assert result2.status == "success", f"Second call failed: {result2.stderr}"
    assert "FILE_NOT_FOUND" not in str(result2.func_return), (
        "File was not found on the second call — sandbox was not persistent"
    )
    assert content in result2.func_return, (
        f"Expected '{content}' in result, got '{result2.func_return}'"
    )


# ---------------------------------------------------------------------------
# LETTA_AGENT_ID injection test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_injects_agent_id_env_var(check_tensorlake_key_is_set, test_user, agent_state):
    """LETTA_AGENT_ID is available as an environment variable inside the sandbox."""

    def get_agent_id() -> str:
        """Returns the LETTA_AGENT_ID environment variable."""
        import os

        return os.getenv("LETTA_AGENT_ID", "NOT_SET")

    tool = create_tool_from_func(get_agent_id)
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)

    sandbox = AsyncToolSandboxTensorlake(
        tool.name,
        {},
        user=test_user,
        tool_id=tool.id,
        agent_id=agent_state.id,
    )
    result = await sandbox.run(agent_state=agent_state)

    assert result.status == "success"
    assert agent_state.id in result.func_return, (
        f"Expected agent_id '{agent_state.id}' in result, got '{result.func_return}'"
    )


# ---------------------------------------------------------------------------
# Additional fixtures for pip / core memory / client injection tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def tool_with_pip_requirements(test_user):
    """Tool that imports numpy and requests — verifies tool-level pip requirements are installed."""

    def use_numpy() -> str:
        """
        Uses numpy to verify pip requirements are installed in the sandbox.

        Returns:
            str: Success message with array sum, or import error.
        """
        try:
            import numpy as np

            arr = np.array([1, 2, 3])
            return f"Success! Array sum: {int(np.sum(arr))}"
        except ImportError as e:
            return f"Import error: {e}"

    tool = create_tool_from_func(use_numpy)
    tool.pip_requirements = [PipRequirement(name="numpy")]
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool



@pytest.fixture
async def list_tools_with_client_tool(test_user):
    """Tool that uses the injected `client` variable to list tools."""
    source_code = '''
def list_tools_via_client() -> str:
    """
    List available tools using the Letta client available in sandbox scope.

    Returns:
        str: Tool count message, or error string.
    """
    if not client:
        return "ERROR: client not available in scope"
    try:
        tools = client.tools.list()
        return f"Found {len([t for t in tools])} tools"
    except Exception as e:
        return f"ERROR: {str(e)}"
'''
    tool = Tool(
        name="list_tools_via_client",
        description="List tools using client available in sandbox scope",
        source_code=source_code,
        source_type="python",
        tool_type=ToolType.CUSTOM,
        json_schema={
            "name": "list_tools_via_client",
            "description": "List tools using client available in sandbox scope",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    tool = await ToolManager().create_or_update_tool_async(tool, test_user)
    yield tool


# ---------------------------------------------------------------------------
# Pip requirements test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_tool_pip_requirements(check_tensorlake_key_is_set, tool_with_pip_requirements, test_user):
    """Tool-level pip requirements are installed before execution."""
    sandbox = AsyncToolSandboxTensorlake(
        tool_with_pip_requirements.name,
        {},
        user=test_user,
        tool_id=tool_with_pip_requirements.id,
        tool_object=tool_with_pip_requirements,
    )
    result = await sandbox.run()
    assert result.status == "success"
    assert "Success!" in result.func_return, f"Expected numpy to be available, got: {result.func_return}"
    assert "Array sum: 6" in result.func_return


# ---------------------------------------------------------------------------
# NOTE: Stateful tool (agent_state round-trip) test is intentionally omitted.
#
# Tools that take `agent_state: "AgentState"` as a parameter (e.g. letta's
# built-in core_memory_replace / core_memory_append) work by pickling the full
# AgentState on the host, deserializing it inside the sandbox, running the tool,
# and pickling the mutated state back.  Deserializing `letta.schemas.agent.AgentState`
# requires the full `letta` package to be importable inside the sandbox.
#
# Installing letta (60+ transitive dependencies, several GB) at sandbox boot time
# is not practical without a pre-built Tensorlake snapshot that has letta already
# baked in.  E2B handles this via its `e2b_sandbox_template_id` setting (a custom
# E2B template image with letta pre-installed).  Tensorlake supports the same
# pattern via `TensorlakeSandboxConfig.snapshot_id` / `TENSORLAKE_SNAPSHOT_ID`
# env var — once a snapshot with letta is created, set that env var and the
# stateful tool test can be added back here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Client injection test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.tensorlake_sandbox
async def test_tensorlake_sandbox_client_injection(check_tensorlake_key_is_set, list_tools_with_client_tool, test_user):
    """
    The sandbox execution script always includes Letta client initialization code.
    Mirrors test_e2b_sandbox_with_client_injection from integration_test_async_tool_sandbox.py.
    Only checks script generation — no live server call needed.
    """
    api_key = os.getenv("LETTA_API_KEY") or "test-key"
    sandbox_env_vars = {
        "LETTA_API_KEY": api_key,
        "LETTA_BASE_URL": os.getenv("LETTA_BASE_URL", "http://localhost:8283"),
    }

    sandbox = AsyncToolSandboxTensorlake(
        tool_name=list_tools_with_client_tool.name,
        args={},
        user=test_user,
        tool_id=list_tools_with_client_tool.id,
        tool_object=list_tools_with_client_tool,
        sandbox_env_vars=sandbox_env_vars,
    )
    await sandbox._init_async()

    assert sandbox.inject_letta_client is True, "Client injection should always be enabled"

    script = await sandbox.generate_execution_script(agent_state=None)
    assert "from letta_client import Letta" in script, "Script should import Letta client"
    assert "LETTA_API_KEY" in script, "Script should check for LETTA_API_KEY"
    assert "client = Letta(" in script or "client = None" in script, "Script should initialise Letta client"
