import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from letta.constants import LETTA_TOOL_EXECUTION_DIR, MODAL_DEFAULT_TIMEOUT
from letta.schemas.agent import AgentState
from letta.schemas.enums import PrimitiveType, SandboxType
from letta.schemas.letta_base import LettaBase, OrmMetadataBase
from letta.schemas.pip_requirement import PipRequirement
from letta.settings import tool_settings

# Sandbox Config


class SandboxRunResult(BaseModel):
    func_return: Optional[Any] = Field(None, description="The function return object")
    agent_state: Optional[AgentState] = Field(None, description="The agent state")
    stdout: Optional[List[str]] = Field(None, description="Captured stdout (e.g. prints, logs) from the function invocation")
    stderr: Optional[List[str]] = Field(None, description="Captured stderr from the function invocation")
    status: Literal["success", "error"] = Field(..., description="The status of the tool execution and return object")
    sandbox_config_fingerprint: str = Field(None, description="The fingerprint of the config for the sandbox")


class LocalSandboxConfig(BaseModel):
    sandbox_dir: Optional[str] = Field(None, description="Directory for the sandbox environment.")
    use_venv: bool = Field(False, description="Whether or not to use the venv, or run directly in the same run loop.")
    venv_name: str = Field(
        "venv",
        description="The name for the venv in the sandbox directory. We first search for an existing venv with this name, otherwise, we make it from the requirements.txt.",
    )
    pip_requirements: List[PipRequirement] = Field(
        default_factory=list,
        description="List of pip packages to install with mandatory name and optional version following semantic versioning. This only is considered when use_venv is True.",
    )

    @property
    def type(self) -> "SandboxType":
        return SandboxType.LOCAL

    @model_validator(mode="before")
    @classmethod
    def set_default_sandbox_dir(cls, data):
        # If `data` is not a dict (e.g., it's another Pydantic model), just return it
        if not isinstance(data, dict):
            return data

        if data.get("sandbox_dir") is None:
            if tool_settings.tool_exec_dir:
                data["sandbox_dir"] = tool_settings.tool_exec_dir
            else:
                data["sandbox_dir"] = LETTA_TOOL_EXECUTION_DIR

        return data


class E2BSandboxConfig(BaseModel):
    timeout: int = Field(5 * 60, description="Time limit for the sandbox (in seconds).")
    template: Optional[str] = Field(None, description="The E2B template id (docker image).")
    pip_requirements: Optional[List[str]] = Field(None, description="A list of pip packages to install on the E2B Sandbox")

    @property
    def type(self) -> "SandboxType":
        return SandboxType.E2B

    @model_validator(mode="before")
    @classmethod
    def set_default_template(cls, data: dict):
        """
        Assign a default template value if the template field is not provided.
        """
        # If `data` is not a dict (e.g., it's another Pydantic model), just return it
        if not isinstance(data, dict):
            return data

        if data.get("template") is None:
            data["template"] = tool_settings.e2b_sandbox_template_id
        return data


class ModalSandboxConfig(BaseModel):
    timeout: int = Field(MODAL_DEFAULT_TIMEOUT, description="Time limit for the sandbox (in seconds).")
    pip_requirements: list[str] | None = Field(None, description="A list of pip packages to install in the Modal sandbox")
    npm_requirements: list[str] | None = Field(None, description="A list of npm packages to install in the Modal sandbox")
    language: Literal["python", "typescript"] = "python"

    @property
    def type(self) -> "SandboxType":
        return SandboxType.MODAL


class TensorlakeSandboxConfig(BaseModel):
    """Configuration for the Tensorlake sandbox provider.

    **Persistence model** — unlike E2B (which creates a fresh sandbox for every tool
    call), Tensorlake sandboxes are *persistent and agent-scoped*:

    - The first tool call for an agent creates a **named** sandbox (``letta-agent-<id>``)
      and caches its ID in memory.
    - Subsequent tool calls for the *same* agent reconnect to the same sandbox.
    - Because the sandbox is **named**, Tensorlake can suspend it when idle and resume
      it on the next ``connect()`` call — installed packages and written files survive
      across tool calls.  Unnamed (ephemeral) sandboxes do **not** support
      suspend/resume and are terminated on timeout.
    - If ``agent_id`` is not provided (e.g. a one-off tool execution), the sandbox is
      created without a name (ephemeral) and terminated immediately after the call —
      same behaviour as E2B.
    - If the sandbox configuration changes (image, memory, pip_requirements, etc.) the
      cached sandbox is evicted and a new one is created automatically.

    This persistent model is useful for agents that accumulate state across multiple
    tool invocations (e.g. writing intermediate files, building up an installed
    environment). If you need a guaranteed clean slate on every call, use E2B instead.
    """

    image: Optional[str] = Field(None, description="Docker image for the sandbox. When None, Tensorlake uses its built-in default image.")
    cpus: float = Field(1.0, description="Number of CPU cores for the sandbox. Memory must stay between 1024–8192 MB per CPU core.")
    memory_mb: int = Field(2048, description="Memory limit for the sandbox (in MB). Must be between 1024–8192 per CPU core.")
    timeout_secs: int = Field(900, description="Time limit for the sandbox (in seconds).")
    allow_internet_access: bool = Field(True, description="Whether to allow internet access from the sandbox.")
    pip_requirements: list[str] | None = Field(None, description="A list of pip packages to pre-install in the sandbox.")
    ephemeral_disk_mb: int = Field(8192, description="Ephemeral disk size for the sandbox (in MB). Installing letta and its transitive dependencies requires several GB.")
    snapshot_id: Optional[str] = Field(None, description="Tensorlake snapshot ID to boot from. When set, letta is assumed pre-installed in the snapshot and will not be re-installed at runtime, eliminating the cold-start pip install cost.")

    @property
    def type(self) -> "SandboxType":
        return SandboxType.TENSORLAKE


class SandboxConfigBase(OrmMetadataBase):
    __id_prefix__ = PrimitiveType.SANDBOX_CONFIG.value


class SandboxConfig(SandboxConfigBase):
    id: str = SandboxConfigBase.generate_id_field()
    type: SandboxType = Field(None, description="The type of sandbox.")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the sandbox.")
    config: Dict = Field(default_factory=lambda: {}, description="The JSON sandbox settings data.")

    def get_e2b_config(self) -> E2BSandboxConfig:
        config_dict = self.config.copy()
        config_dict["template"] = tool_settings.e2b_sandbox_template_id
        return E2BSandboxConfig(**config_dict)

    def get_local_config(self) -> LocalSandboxConfig:
        return LocalSandboxConfig(**self.config)

    def get_modal_config(self) -> ModalSandboxConfig:
        return ModalSandboxConfig(**self.config)

    def get_tensorlake_config(self) -> TensorlakeSandboxConfig:
        config_dict = self.config.copy()
        # Overlay the current env var so that setting TENSORLAKE_SNAPSHOT_ID takes
        # effect even for orgs whose default config was created before the var was set
        # (mirrors how get_e2b_config() overlays e2b_sandbox_template_id).
        if tool_settings.tensorlake_snapshot_id:
            config_dict["snapshot_id"] = tool_settings.tensorlake_snapshot_id
        return TensorlakeSandboxConfig(**config_dict)

    def fingerprint(self) -> str:
        # Only take into account type, org_id, and the config items
        # Canonicalize input data into JSON with sorted keys
        hash_input = json.dumps(
            {
                "type": self.type.value,
                "organization_id": self.organization_id,
                "config": self.config,
            },
            sort_keys=True,  # Ensure stable ordering
            separators=(",", ":"),  # Minimize serialization differences
        )

        # Compute SHA-256 hash
        hash_digest = hashlib.sha256(hash_input.encode("utf-8")).digest()

        # Convert the digest to an integer for compatibility with Python's hash requirements
        return str(int.from_bytes(hash_digest, byteorder="big"))


class SandboxConfigCreate(LettaBase):
    config: Union[LocalSandboxConfig, E2BSandboxConfig, ModalSandboxConfig, TensorlakeSandboxConfig] = Field(
        ..., description="The configuration for the sandbox."
    )


class SandboxConfigUpdate(LettaBase):
    """Pydantic model for updating SandboxConfig fields."""

    config: Union[LocalSandboxConfig, E2BSandboxConfig, ModalSandboxConfig, TensorlakeSandboxConfig] = Field(
        None, description="The JSON configuration data for the sandbox."
    )
