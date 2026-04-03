import asyncio
import contextlib
import time
from typing import Any, ClassVar, Dict, Optional

from letta.log import get_logger
from letta.otel.tracing import log_event, trace_method
from letta.schemas.agent import AgentState
from letta.schemas.enums import SandboxType
from letta.schemas.sandbox_config import SandboxConfig, TensorlakeSandboxConfig
from letta.schemas.tool import Tool
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.services.helpers.tool_parser_helper import parse_stdout_best_effort
from letta.services.tool_sandbox.base import AsyncToolSandboxBase
from letta.settings import tool_settings
from letta.types import JsonDict
from letta.utils import get_friendly_error_msg

logger = get_logger(__name__)


class AsyncToolSandboxTensorlake(AsyncToolSandboxBase):
    METADATA_CONFIG_STATE_KEY = "config_state"
    _FINGERPRINT_PATH = "/tmp/_letta_env_fingerprint"

    # In-memory map of agent_id -> (sandbox_id, env_fingerprint) so we reconnect
    # to the same persistent workspace and can detect stale base environments.
    _agent_sandbox_ids: ClassVar[Dict[str, tuple]] = {}
    # Per-agent asyncio locks to serialize sandbox creation. Without these, two
    # concurrent tool calls for the same agent could both see an empty cache entry,
    # both create a new sandbox, and the second write would overwrite the first —
    # leaking a sandbox and making persistence nondeterministic.
    _agent_locks: ClassVar[Dict[str, asyncio.Lock]] = {}

    def __init__(
        self,
        tool_name: str,
        args: JsonDict,
        user,
        tool_id: str,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        tool_object: Optional[Tool] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        sandbox_env_vars: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            tool_name,
            args,
            user,
            tool_id=tool_id,
            agent_id=agent_id,
            project_id=project_id,
            tool_object=tool_object,
            sandbox_config=sandbox_config,
            sandbox_env_vars=sandbox_env_vars,
        )

    async def generate_execution_script(self, agent_state: Optional[AgentState] = None, wrap_print_with_markers: bool = False) -> str:
        """Override to wrap the base64 result expression in print().

        The base class emits a bare expression for the result when
        wrap_print_with_markers=False (designed for E2B's Jupyter kernel which
        auto-captures expression values).  Tensorlake runs plain ``python3 script.py``,
        so bare expressions produce no output.  We wrap the last line in ``print()``
        to ensure the result lands in stdout.
        """
        code = await super().generate_execution_script(agent_state=agent_state, wrap_print_with_markers=False)
        # The last non-empty line is the bare base64 expression. Wrap it in print().
        lines = code.rstrip("\n").split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                lines[i] = f"print({lines[i]})"
                break
        return "\n".join(lines) + "\n"

    @trace_method
    async def run(
        self,
        agent_state: Optional[AgentState] = None,
        additional_env_vars: Optional[Dict] = None,
    ) -> ToolExecutionResult:
        await self._init_async()

        if self.provided_sandbox_config:
            sbx_config = self.provided_sandbox_config
        else:
            sbx_config = await self.sandbox_config_manager.get_or_create_default_sandbox_config_async(
                sandbox_type=SandboxType.TENSORLAKE, actor=self.user
            )

        if self.is_typescript_tool():
            raise NotImplementedError(
                "Tensorlake sandbox does not support TypeScript tools. "
                "Use E2B or Modal sandbox for TypeScript tool execution."
            )

        tl_config = sbx_config.get_tensorlake_config()
        sandbox = None
        try:
            sandbox = await self._get_or_create_sandbox(tl_config)
            envs = await self._gather_env_vars(agent_state, additional_env_vars, sbx_config.id, is_local=False)
            code = await self.generate_execution_script(agent_state=agent_state)
            logger.info(f"Tensorlake execution started for sandbox {sandbox.sandbox_id}: {self.tool_name}")
            log_event(
                "tensorlake_execution_started",
                {
                    "tool": self.tool_name,
                    "sandbox_id": sandbox.sandbox_id,
                    "agent_id": self.agent_id,
                },
            )

            start_time = time.perf_counter()
            try:
                result = await asyncio.to_thread(self._run_code_sync, sandbox, code, envs, tl_config.timeout_secs)
            except asyncio.CancelledError:
                execution_time = time.perf_counter() - start_time
                logger.info(
                    f"Tensorlake execution cancelled for sandbox {sandbox.sandbox_id}: "
                    f"{self.tool_name} (took {execution_time:.2f}s)"
                )
                log_event(
                    "tensorlake_execution_cancelled",
                    {"tool": self.tool_name, "sandbox_id": sandbox.sandbox_id, "execution_time_seconds": execution_time},
                )
                raise Exception("Execution cancelled. Transient failure, please retry.")

            execution_time = time.perf_counter() - start_time
            logger.info(
                f"Tensorlake execution completed in {execution_time:.2f}s "
                f"for sandbox {sandbox.sandbox_id}, tool: {self.tool_name}"
            )

            result_b64 = result["result_b64"]
            stdout_lines = result["stdout"].splitlines() if result["stdout"] else []
            stderr_lines = result["stderr"].splitlines() if result["stderr"] else []
            exit_code = result["exit_code"]

            if exit_code != 0:
                error_msg = "\n".join(stderr_lines) if stderr_lines else f"exit code {exit_code}"
                func_return = get_friendly_error_msg(
                    function_name=self.tool_name,
                    exception_name="SandboxExecutionError",
                    exception_message=error_msg,
                )
                logger.warning(
                    f"Tensorlake execution failed for sandbox {sandbox.sandbox_id}: "
                    f"{self.tool_name} (took {execution_time:.2f}s)"
                )
                log_event(
                    "tensorlake_execution_failed",
                    {
                        "tool": self.tool_name,
                        "sandbox_id": sandbox.sandbox_id,
                        "error": error_msg,
                        "execution_time_seconds": execution_time,
                    },
                )
                return ToolExecutionResult(
                    func_return=func_return,
                    agent_state=None,
                    stdout=stdout_lines,
                    stderr=stderr_lines,
                    status="error",
                    sandbox_config_fingerprint=sbx_config.fingerprint(),
                )

            if not result_b64:
                logger.warning(
                    f"Tensorlake execution empty for sandbox {sandbox.sandbox_id}: "
                    f"{self.tool_name} (took {execution_time:.2f}s) — no result line in stdout"
                )
                raise ValueError(f"Tool {self.tool_name} returned no output")

            # result_b64 is the last stdout line: the base64-encoded pickle of {results, agent_state}.
            func_return, updated_agent_state = parse_stdout_best_effort(result_b64)

            log_event(
                "tensorlake_execution_succeeded",
                {
                    "tool": self.tool_name,
                    "sandbox_id": sandbox.sandbox_id,
                    "func_return": func_return,
                    "execution_time_seconds": execution_time,
                },
            )

            return ToolExecutionResult(
                func_return=func_return,
                agent_state=updated_agent_state,
                stdout=stdout_lines,
                stderr=stderr_lines,
                status="success",
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )

        except Exception as e:
            logger.error(f"Tensorlake sandbox execution raised an unexpected error: {e}")
            func_return = get_friendly_error_msg(
                function_name=self.tool_name,
                exception_name=type(e).__name__,
                exception_message=str(e),
            )
            return ToolExecutionResult(
                func_return=func_return,
                agent_state=None,
                stdout=[],
                stderr=[str(e)],
                status="error",
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )
        finally:
            # Non-agent executions have no persistent identity to reconnect to, so
            # terminate immediately to avoid accumulating suspended sandboxes.
            if sandbox is not None and not self.agent_id:
                try:
                    await asyncio.to_thread(sandbox.terminate)
                except Exception as term_err:
                    logger.warning(f"Failed to terminate ephemeral Tensorlake sandbox {sandbox.sandbox_id}: {term_err}")
        # Agent-scoped sandboxes are intentionally kept alive.  Because they are created
        # with a name (``letta-agent-<id>``), Tensorlake suspends them when idle and
        # resumes them on the next connect() call.

    @staticmethod
    def _run_code_sync(sandbox, code: str, envs: dict, timeout_secs: int) -> dict:
        """
        Write Python code to a unique temp file inside the sandbox and execute it.

        Each invocation uses a unique file path so that parallel tool calls on the
        same agent-scoped sandbox do not overwrite each other's scripts.

        Returns a dict with:
          - result_b64: the last non-empty stdout line (the base64-encoded pickle)
          - stdout: all other stdout lines (tool print() output)
          - stderr: stderr lines
          - exit_code: process exit code
        """
        import uuid

        exec_path = f"/tmp/_letta_tool_exec_{uuid.uuid4().hex[:12]}.py"
        sandbox.write_file(exec_path, code.encode())
        run_env = {k: str(v) for k, v in envs.items()} if envs else {}
        # Packages are installed via --target=/tmp/letta_packages; prepend to PYTHONPATH
        # so Python finds them without requiring system-level installation.
        # We prepend (rather than setdefault) so that any caller-supplied PYTHONPATH is
        # preserved — both paths remain importable.
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = "/tmp/letta_packages:" + existing_pythonpath if existing_pythonpath else "/tmp/letta_packages"
        result = sandbox.run(
            command="python3",
            args=[exec_path],
            env=run_env,
            timeout=timeout_secs,
        )
        # Clean up the per-invocation script to avoid leaking files in persistent
        # agent-scoped sandboxes.  Best-effort — don't let cleanup failure mask the
        # actual tool result.
        try:
            sandbox.delete_file(exec_path)
        except Exception:
            pass
        all_lines = result.stdout.splitlines() if result.stdout else []
        # The last non-empty line is the base64-encoded pickle result.
        # All preceding lines are tool print() output.
        non_empty = [l for l in all_lines if l.strip()]
        result_b64 = non_empty[-1] if non_empty else ""
        # Strip the result line from user-visible stdout so callers don't see the
        # raw base64 payload mixed in with tool print() output.
        if result_b64:
            # Find the last occurrence of the result line and exclude it.
            for i in range(len(all_lines) - 1, -1, -1):
                if all_lines[i].strip() == result_b64:
                    user_stdout_lines = all_lines[:i] + all_lines[i + 1 :]
                    break
            else:
                user_stdout_lines = all_lines
        else:
            user_stdout_lines = all_lines
        return {
            "result_b64": result_b64,
            "stdout": "\n".join(user_stdout_lines) if user_stdout_lines else "",
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    async def _get_or_create_sandbox(self, tl_config: TensorlakeSandboxConfig):
        """
        Connect to an existing agent-scoped sandbox or create a new one.

        Persistence strategy:
          - We keep an in-memory map of agent_id -> (sandbox_id, env_fingerprint).
          - The fingerprint covers the letta version, docker image, and config-level
            pip requirements.  If any of these change between calls the cached sandbox
            is discarded and a fresh one is created.
          - On reconnect we only install tool-level requirements that may differ from
            the previous tool call.
          - Agent-scoped sandboxes are created with a ``name`` so that Tensorlake can
            suspend them when idle and resume them on the next ``connect()`` call.
            Only **named** sandboxes support suspend/resume — ephemeral (unnamed)
            sandboxes run until timeout or manual termination and cannot be resumed.
          - A per-agent asyncio.Lock serializes concurrent calls so that only one
            coroutine at a time can create or reconnect to the sandbox for a given
            agent, preventing duplicate workspace creation under load.
        """
        if self.agent_id:
            lock = self.__class__._agent_locks.setdefault(self.agent_id, asyncio.Lock())
        else:
            lock = None

        async with (lock if lock is not None else contextlib.nullcontext()):
            return await self._get_or_create_sandbox_locked(tl_config)

    async def _get_or_create_sandbox_locked(self, tl_config: TensorlakeSandboxConfig):
        """Inner implementation — always called while holding the per-agent lock (or with no lock for ephemeral calls)."""
        from tensorlake.sandbox import SandboxClient, SandboxNotFoundError

        api_key = tool_settings.tensorlake_api_key
        client = SandboxClient(api_key=api_key)

        # Resolve letta version once; used for both pinning and fingerprinting.
        try:
            from importlib.metadata import version as _pkg_version

            letta_version = _pkg_version("letta")
            letta_pin = f"letta=={letta_version}"
        except Exception:
            letta_version = "unknown"
            letta_pin = "letta"

        current_fingerprint = self._env_fingerprint(tl_config, letta_version)

        # Check whether the cached sandbox is still valid for the current environment.
        cached = self._agent_sandbox_ids.get(self.agent_id) if self.agent_id else None
        existing_id = cached[0] if cached else None
        stored_fingerprint = cached[1] if cached else None

        if existing_id and stored_fingerprint != current_fingerprint:
            # Evict the stale mapping. We intentionally do not call client.delete()
            # here: concurrent tool calls within the same agent share the same config,
            # so a fingerprint mismatch only occurs across separate requests.  Deleting
            # from the hot path would race with any in-flight execution using the old
            # sandbox.  Tensorlake's built-in timeout_secs will reclaim it automatically.
            logger.info(
                f"Tensorlake sandbox env changed for agent {self.agent_id} "
                f"(fingerprint {stored_fingerprint!r} -> {current_fingerprint!r}); evicting sandbox {existing_id}."
            )
            del self._agent_sandbox_ids[self.agent_id]
            existing_id = None

        # Fallback: if the in-memory cache is empty (e.g. after server restart) but we
        # have an agent_id, look up the named sandbox via the Tensorlake API.  Named
        # sandboxes survive process restarts — we just need to rediscover their ID.
        # To guard against config drift (image/memory/cpus changed between restarts),
        # we store the fingerprint as a file inside the sandbox at creation time and
        # validate it on rediscovery.  If it doesn't match, the old sandbox is
        # terminated and a fresh one is created with the current config.
        if not existing_id and self.agent_id:
            sandbox_name = f"letta-agent-{self.agent_id}"
            try:
                from tensorlake.sandbox import SandboxStatus

                all_sandboxes = await asyncio.to_thread(client.list)
                for info in all_sandboxes:
                    if info.name == sandbox_name and info.status != SandboxStatus.TERMINATED:
                        # Validate the fingerprint stored inside the sandbox.
                        try:
                            tmp_sbx = await asyncio.to_thread(client.connect, info.sandbox_id)
                            stored_fp = await asyncio.to_thread(tmp_sbx.read_file, self._FINGERPRINT_PATH)
                            stored_fp = stored_fp.decode("utf-8").strip()
                        except Exception:
                            stored_fp = None

                        if stored_fp == current_fingerprint:
                            existing_id = info.sandbox_id
                            logger.info(
                                f"Tensorlake sandbox rediscovered by name '{sandbox_name}': {existing_id} "
                                f"(fingerprint validated)"
                            )
                            log_event("tensorlake_sandbox_rediscovered", {"sandbox_id": existing_id, "name": sandbox_name})
                        else:
                            logger.info(
                                f"Tensorlake sandbox '{sandbox_name}' ({info.sandbox_id}) has stale config "
                                f"(fingerprint {stored_fp!r} != {current_fingerprint!r}); terminating."
                            )
                            log_event("tensorlake_sandbox_terminated_stale", {"sandbox_id": info.sandbox_id})
                            try:
                                await asyncio.to_thread(client.delete, info.sandbox_id)
                            except Exception:
                                pass
                        break
            except Exception as e:
                logger.warning(f"Failed to list Tensorlake sandboxes for name lookup: {e}")

        log_event(
            "tensorlake_sandbox_connect",
            {
                "existing_sandbox_id": existing_id,
                "agent_id": self.agent_id,
                "image": tl_config.image,
                "memory_mb": tl_config.memory_mb,
            },
        )

        sandbox = None

        if existing_id:
            try:
                sandbox = await asyncio.to_thread(client.connect, existing_id)
                logger.info(f"Tensorlake sandbox resumed: {existing_id} (agent_id={self.agent_id})")
                log_event("tensorlake_sandbox_resumed", {"sandbox_id": existing_id})
                # Re-populate the in-memory cache so subsequent calls skip the list() lookup.
                if self.agent_id:
                    self.__class__._agent_sandbox_ids[self.agent_id] = (existing_id, current_fingerprint)
            except SandboxNotFoundError:
                logger.warning(
                    f"Tensorlake sandbox {existing_id} no longer exists for agent {self.agent_id}; creating new one."
                )
                if self.agent_id:
                    self.__class__._agent_sandbox_ids.pop(self.agent_id, None)
                sandbox = None

        if sandbox is None:
            # Named sandboxes support suspend/resume, which is required for the
            # agent-scoped persistence model (files and packages surviving across
            # tool calls).  Ephemeral (unnamed) sandboxes cannot be suspended and
            # are terminated on timeout — they must NOT be used for agent-scoped
            # sandboxes.  For one-off executions (no agent_id) we intentionally
            # omit the name so the sandbox is ephemeral.
            sandbox_name = f"letta-agent-{self.agent_id}" if self.agent_id else None
            sandbox = await asyncio.to_thread(
                client.create_and_connect,
                image=tl_config.image,
                cpus=tl_config.cpus,
                memory_mb=tl_config.memory_mb,
                ephemeral_disk_mb=tl_config.ephemeral_disk_mb,
                timeout_secs=tl_config.timeout_secs,
                allow_internet_access=tl_config.allow_internet_access,
                snapshot_id=tl_config.snapshot_id or None,
                name=sandbox_name,
            )
            logger.info(f"Tensorlake sandbox created: {sandbox.sandbox_id} (agent_id={self.agent_id}, snapshot={tl_config.snapshot_id})")
            log_event("tensorlake_sandbox_created", {"sandbox_id": sandbox.sandbox_id, "snapshot_id": tl_config.snapshot_id})

            # pydantic is always needed (coerces tool arguments).
            # letta is needed when the tool takes agent_state as a parameter —
            # the generated script adds `import letta; from letta import *` in that case,
            # and pickle deserialization of AgentState requires letta to be importable.
            # Installing letta's full dep tree (~60 packages, several GB) at boot time
            # is expensive.  The recommended approach is to create a Tensorlake snapshot
            # with letta pre-installed and set TensorlakeSandboxConfig.snapshot_id (or
            # the TENSORLAKE_SNAPSHOT_ID env var).  When a snapshot is used, letta is
            # already present and the install step is skipped entirely.
            # pydantic: used for argument coercion in the generated script.
            # letta-client: needed for the injected `client` variable.
            # packaging: imported by the generated script to detect letta-client version.
            base_requirements = ["pydantic", "letta-client", "packaging"]
            if self.inject_agent_state and not tl_config.snapshot_id:
                base_requirements.append(letta_pin)
            tool_requirements = [str(req) for req in self.tool.pip_requirements] if self.tool and self.tool.pip_requirements else []
            all_requirements = base_requirements + (tl_config.pip_requirements or []) + tool_requirements
            try:
                await self._install_packages(sandbox, all_requirements)
            except RuntimeError:
                # Bootstrap failed — clean up so we don't cache a broken sandbox.
                try:
                    await asyncio.to_thread(sandbox.terminate)
                except Exception:
                    pass
                raise

            # Write the environment fingerprint into the sandbox so it can be
            # validated on rediscovery after a server restart (see name-based
            # lookup above).
            try:
                await asyncio.to_thread(
                    sandbox.write_file, self._FINGERPRINT_PATH, current_fingerprint.encode("utf-8")
                )
            except Exception as e:
                logger.warning(f"Failed to write fingerprint to sandbox {sandbox.sandbox_id}: {e}")

            # Only cache the sandbox after successful bootstrap.
            if self.agent_id:
                self.__class__._agent_sandbox_ids[self.agent_id] = (sandbox.sandbox_id, current_fingerprint)
        else:
            # Reconnected to an existing sandbox. Base packages are already installed
            # and validated via fingerprint. Install any tool-level deps that may not
            # yet be present (pip install is a no-op for already-installed packages).
            # Also install letta if this call requires agent_state injection and no
            # snapshot was used — the sandbox may have been created for a stateless
            # tool that only got pydantic, but now needs letta for pickle round-trip.
            extra_requirements = []
            if self.inject_agent_state and not tl_config.snapshot_id:
                extra_requirements.append(letta_pin)
            tool_requirements = [str(req) for req in self.tool.pip_requirements] if self.tool and self.tool.pip_requirements else []
            all_reconnect_requirements = extra_requirements + tool_requirements
            if all_reconnect_requirements:
                try:
                    await self._install_packages(sandbox, all_reconnect_requirements)
                except RuntimeError:
                    # Partially mutated — evict and terminate so the next call gets a fresh sandbox.
                    if self.agent_id:
                        del self.__class__._agent_sandbox_ids[self.agent_id]
                    try:
                        await asyncio.to_thread(sandbox.terminate)
                    except Exception:
                        pass
                    raise

        return sandbox

    @staticmethod
    def _env_fingerprint(tl_config: TensorlakeSandboxConfig, letta_version: str) -> str:
        """Short hash covering all sandbox creation parameters so stale sandboxes are detected."""
        import hashlib
        import json

        data = json.dumps(
            {
                "letta_version": letta_version,
                "image": tl_config.image,
                "cpus": tl_config.cpus,
                "memory_mb": tl_config.memory_mb,
                "ephemeral_disk_mb": tl_config.ephemeral_disk_mb,
                "timeout_secs": tl_config.timeout_secs,
                "allow_internet_access": tl_config.allow_internet_access,
                "pip_requirements": sorted(tl_config.pip_requirements or []),
                "snapshot_id": tl_config.snapshot_id,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def _install_packages(self, sandbox, packages: list) -> None:
        """Install pip packages into the sandbox, raising RuntimeError on any failure.

        Strategy: install everything to /tmp/letta_packages via --target so we bypass
        PEP 668 and non-root filesystem restrictions.
        """
        import shlex

        for package in packages:
            # Install to an explicit writable directory to bypass PEP 668 and non-root
            # filesystem restrictions.  PYTHONPATH=/tmp/letta_packages is set at
            # execution time so Python finds the installed packages.
            cmd = (
                "export HOME=/tmp TMPDIR=/tmp && "
                f"python3 -m pip install --target=/tmp/letta_packages "
                f"--prefer-binary --no-cache-dir {shlex.quote(package)}"
            )
            result = await asyncio.to_thread(
                sandbox.run,
                command="sh",
                args=["-c", cmd],
                working_dir="/tmp",
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to install '{package}' in Tensorlake sandbox "
                    f"{sandbox.sandbox_id}: {result.stderr}"
                )
