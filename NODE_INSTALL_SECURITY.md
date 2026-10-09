# Maintainer Node installation

Owned tool installs use pnpm 11.28.5, a strict 24-hour release delay, required
publication timestamps and disabled install hooks. The fixed manager is
bootstrapped with npm hooks disabled. Requested tool versions are still
supported through these checks; releases younger than the delay fail closed.

Fern follows its existing configured version where present. Default Pi and
image Codex versions are pinned to already-aged releases. User workload
package-manager commands and npm publication remain supported. Full Linux
image/tool startup verification is required before merging this migration.
