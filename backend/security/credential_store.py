from __future__ import annotations

import ctypes
import json
import os
import re
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
ERROR_NOT_FOUND = 1168
_MAX_CREDENTIAL_BLOB_BYTES = 5 * 512

CTRADER_SECRET_ENV_NAMES = (
    "CTRADER_CLIENT_ID",
    "CTRADER_CLIENT_SECRET",
    "CTRADER_ACCESS_TOKEN",
)
_CREDENTIAL_TARGETS = {
    "CTRADER_CLIENT_ID": "TradeAgent/cTrader/client_id",
    "CTRADER_CLIENT_SECRET": "TradeAgent/cTrader/client_secret",
    "CTRADER_ACCESS_TOKEN": "TradeAgent/cTrader/access_token",
}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"^\\s*(?:export\\s+)?("
    + "|".join(re.escape(name) for name in CTRADER_SECRET_ENV_NAMES)
    + r")\\s*="
)


class CredentialStoreError(RuntimeError):
    """Raised when Windows Credential Manager cannot safely read/write a secret."""


@dataclass(frozen=True)
class SecretValue:
    value: str | None
    source: str


class _CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


_PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)


def _is_windows() -> bool:
    return os.name == "nt"


def _target_for(env_name: str) -> str:
    try:
        return _CREDENTIAL_TARGETS[env_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported cTrader credential: {env_name}") from exc


def _windows_api():
    if not _is_windows():
        raise CredentialStoreError("Windows Credential Manager is only available on Windows.")

    advapi32 = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_PCREDENTIALW),
    ]
    advapi32.CredReadW.restype = wintypes.BOOL
    advapi32.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
    advapi32.CredWriteW.restype = wintypes.BOOL
    advapi32.CredDeleteW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    advapi32.CredDeleteW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [ctypes.c_void_p]
    advapi32.CredFree.restype = None
    return advapi32


def _read_windows_generic_credential(target: str) -> str | None:
    api = _windows_api()
    credential_ptr = _PCREDENTIALW()

    if not api.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(credential_ptr)):
        error = ctypes.get_last_error()
        if error == ERROR_NOT_FOUND:
            return None
        raise CredentialStoreError(
            f"Windows Credential Manager read failed for {target!r} (error {error})."
        )

    try:
        credential = credential_ptr.contents
        raw = ctypes.string_at(
            credential.CredentialBlob,
            int(credential.CredentialBlobSize),
        )
        return raw.decode("utf-8")
    finally:
        api.CredFree(credential_ptr)


def _write_windows_generic_credential(target: str, value: str) -> None:
    encoded = value.encode("utf-8")
    if not encoded:
        raise ValueError("Credential value must not be empty.")
    if len(encoded) > _MAX_CREDENTIAL_BLOB_BYTES:
        raise ValueError("Credential value is too large for Windows Credential Manager.")

    blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
    credential = _CREDENTIALW()
    credential.Type = CRED_TYPE_GENERIC
    credential.TargetName = target
    credential.CredentialBlobSize = len(encoded)
    credential.CredentialBlob = ctypes.cast(
        blob,
        ctypes.POINTER(ctypes.c_ubyte),
    )
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = "TradeAgent"

    api = _windows_api()
    if not api.CredWriteW(ctypes.byref(credential), 0):
        error = ctypes.get_last_error()
        raise CredentialStoreError(
            f"Windows Credential Manager write failed for {target!r} (error {error})."
        )


def _delete_windows_generic_credential(target: str) -> bool:
    api = _windows_api()
    if api.CredDeleteW(target, CRED_TYPE_GENERIC, 0):
        return True

    error = ctypes.get_last_error()
    if error == ERROR_NOT_FOUND:
        return False
    raise CredentialStoreError(
        f"Windows Credential Manager delete failed for {target!r} (error {error})."
    )


def read_windows_credential(env_name: str) -> str | None:
    return _read_windows_generic_credential(_target_for(env_name))


def write_windows_credential(env_name: str, value: str) -> None:
    _write_windows_generic_credential(_target_for(env_name), value)


def delete_windows_credential(env_name: str) -> bool:
    return _delete_windows_generic_credential(_target_for(env_name))


def get_ctrader_secret_with_source(
    env_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> SecretValue:
    _target_for(env_name)

    if _is_windows():
        value = read_windows_credential(env_name)
        if value:
            return SecretValue(
                value=value,
                source="windows_credential_manager",
            )

    source_env = os.environ if environ is None else environ
    value = source_env.get(env_name)
    if value:
        return SecretValue(value=value, source="environment")
    return SecretValue(value=None, source="missing")


def get_ctrader_secret(
    env_name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    return get_ctrader_secret_with_source(env_name, environ=environ).value


def read_ctrader_secrets_from_env_file(path: Path) -> dict[str, str]:
    values = dotenv_values(path)
    secrets: dict[str, str] = {}
    missing: list[str] = []

    for name in CTRADER_SECRET_ENV_NAMES:
        value = values.get(name)
        if value is None or not str(value).strip():
            missing.append(name)
        else:
            secrets[name] = str(value)

    if missing:
        raise CredentialStoreError(
            "The environment file is missing required cTrader credential values: "
            + ", ".join(missing)
        )
    return secrets


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f"{path.name}.tradeagent-tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def scrub_ctrader_secrets_from_env_file(path: Path) -> int:
    original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept_lines = [
        line for line in original_lines if _SECRET_ASSIGNMENT_RE.match(line) is None
    ]
    removed = len(original_lines) - len(kept_lines)

    if removed:
        _atomic_write_text(path, "".join(kept_lines))
    return removed


def restore_ctrader_secrets_to_env_file(path: Path) -> None:
    secrets = {
        name: read_windows_credential(name)
        for name in CTRADER_SECRET_ENV_NAMES
    }
    missing = [name for name, value in secrets.items() if not value]
    if missing:
        raise CredentialStoreError(
            "Windows Credential Manager is missing required cTrader credentials: "
            + ", ".join(missing)
        )

    existing_lines = (
        path.read_text(encoding="utf-8").splitlines(keepends=True)
        if path.exists()
        else []
    )
    seen: set[str] = set()
    output: list[str] = []

    for line in existing_lines:
        match = _SECRET_ASSIGNMENT_RE.match(line)
        if match is None:
            output.append(line)
            continue

        name = match.group(1)
        output.append(f"{name}={json.dumps(secrets[name])}\n")
        seen.add(name)

    if output and not output[-1].endswith("\n"):
        output[-1] += "\n"
    if output and output[-1].strip():
        output.append("\n")

    for name in CTRADER_SECRET_ENV_NAMES:
        if name not in seen:
            output.append(f"{name}={json.dumps(secrets[name])}\n")

    _atomic_write_text(path, "".join(output))


def migrate_ctrader_secrets_from_env_file(
    path: Path,
    *,
    scrub_env: bool = False,
) -> None:
    if not _is_windows():
        raise CredentialStoreError(
            "cTrader credential migration requires Windows Credential Manager."
        )

    secrets = read_ctrader_secrets_from_env_file(path)
    previous = {
        name: read_windows_credential(name)
        for name in CTRADER_SECRET_ENV_NAMES
    }

    try:
        for name, value in secrets.items():
            write_windows_credential(name, value)

        failed_verification = [
            name
            for name, value in secrets.items()
            if read_windows_credential(name) != value
        ]
        if failed_verification:
            raise CredentialStoreError(
                "Credential write verification failed for: "
                + ", ".join(failed_verification)
            )
    except Exception:
        for name, old_value in previous.items():
            if old_value is None:
                delete_windows_credential(name)
            else:
                write_windows_credential(name, old_value)
        raise

    if scrub_env:
        scrub_ctrader_secrets_from_env_file(path)


def ctrader_credential_status(path: Path) -> dict[str, dict[str, str | bool]]:
    env_values = dotenv_values(path) if path.exists() else {}
    result: dict[str, dict[str, str | bool]] = {}

    for name in CTRADER_SECRET_ENV_NAMES:
        env_present = bool(env_values.get(name))
        vault_present = False
        if _is_windows():
            vault_present = bool(read_windows_credential(name))

        if vault_present:
            effective_source = "windows_credential_manager"
        elif env_present:
            effective_source = "env_file"
        else:
            effective_source = "missing"

        result[name] = {
            "windows_credential_manager": vault_present,
            "env_file": env_present,
            "effective_source": effective_source,
        }

    return result
