from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_api_keys_file(
    path: Path | str | None,
    *,
    required: bool = False,
) -> Path | None:
    """Load NAME=value API keys from a local text file into missing env vars."""
    if path is None:
        return None

    api_keys_file = Path(path)
    if not api_keys_file.exists():
        if required:
            raise RuntimeError(f"API keys file not found: {api_keys_file}")
        return None
    if not api_keys_file.is_file():
        raise RuntimeError(f"API keys path is not a file: {api_keys_file}")

    for line_number, raw_line in enumerate(
        api_keys_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            # Bare-value key files (e.g. ~/api-keys/exa-api-key.txt hold just the
            # token, no NAME=) are unmappable to an env var here; skip with a warning
            # instead of crashing --api-keys-file. Use NAME=value or `export NAME=...`.
            logger.warning(
                "API keys file line %d has no NAME=value (bare value?); skipped. "
                "Use NAME=value or export the key as an env var.",
                line_number,
            )
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME_RE.fullmatch(name):
            raise RuntimeError(
                f"Invalid API keys file variable name on line {line_number}: {name}"
            )
        value = _unquote_value(value.strip())
        if value and not os.environ.get(name):
            os.environ[name] = value
    return api_keys_file


def _unquote_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def resolve_api_key(env_name: str | None, *, key_source: str = "env") -> str | None:
    """Single seam for resolving a source's API key, so the key source is swappable.

    - ``env`` (default, BYO-key tier): read the key from the environment (which
      may have been populated from a local api-keys file).
    - ``proxy`` (hosted tier): the deployment fronts the API with a proxy that
      injects auth upstream, so the client legitimately holds no key. Point the
      source's ``base_url`` at the proxy and this returns None.
    """
    if key_source == "proxy":
        return None
    if not env_name:
        return None
    return os.environ.get(env_name)
