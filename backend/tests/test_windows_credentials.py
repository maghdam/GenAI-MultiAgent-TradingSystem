from pathlib import Path

import pytest

from backend.security import credential_store as store


def test_windows_credential_manager_precedes_environment(monkeypatch) -> None:
    monkeypatch.setattr(store, "_is_windows", lambda: True)
    monkeypatch.setattr(
        store,
        "read_windows_credential",
        lambda name: "vault-value" if name == "CTRADER_ACCESS_TOKEN" else None,
    )

    result = store.get_ctrader_secret_with_source(
        "CTRADER_ACCESS_TOKEN",
        environ={"CTRADER_ACCESS_TOKEN": "env-value"},
    )

    assert result.value == "vault-value"
    assert result.source == "windows_credential_manager"


def test_environment_is_fallback_when_vault_entry_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(store, "_is_windows", lambda: True)
    monkeypatch.setattr(store, "read_windows_credential", lambda _name: None)

    result = store.get_ctrader_secret_with_source(
        "CTRADER_CLIENT_SECRET",
        environ={"CTRADER_CLIENT_SECRET": "env-value"},
    )

    assert result.value == "env-value"
    assert result.source == "environment"


def test_non_windows_uses_environment_without_touching_windows_store(monkeypatch) -> None:
    monkeypatch.setattr(store, "_is_windows", lambda: False)

    def fail_if_called(_name: str):
        raise AssertionError("Windows Credential Manager should not be queried")

    monkeypatch.setattr(store, "read_windows_credential", fail_if_called)

    result = store.get_ctrader_secret_with_source(
        "CTRADER_CLIENT_ID",
        environ={"CTRADER_CLIENT_ID": "client-id"},
    )

    assert result.value == "client-id"
    assert result.source == "environment"


def test_missing_secret_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(store, "_is_windows", lambda: False)

    result = store.get_ctrader_secret_with_source(
        "CTRADER_ACCESS_TOKEN",
        environ={},
    )

    assert result.value is None
    assert result.source == "missing"


def test_unknown_secret_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported cTrader credential"):
        store.get_ctrader_secret("CTRADER_PASSWORD", environ={})


def test_scrub_removes_only_ctrader_secrets(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "CTRADER_CLIENT_ID=id",
                "CTRADER_CLIENT_SECRET=secret",
                "CTRADER_ACCESS_TOKEN=token",
                "CTRADER_HOST_TYPE=demo",
                "CTRADER_ACCOUNT_ID=44089601",
                "OLLAMA_URL=http://127.0.0.1:11434",
                "",
            ]
        ),
        encoding="utf-8",
    )

    removed = store.scrub_ctrader_secrets_from_env_file(env_path)
    contents = env_path.read_text(encoding="utf-8")

    assert removed == 3
    assert "CTRADER_CLIENT_ID=" not in contents
    assert "CTRADER_CLIENT_SECRET=" not in contents
    assert "CTRADER_ACCESS_TOKEN=" not in contents
    assert "CTRADER_HOST_TYPE=demo" in contents
    assert "CTRADER_ACCOUNT_ID=44089601" in contents
    assert "OLLAMA_URL=http://127.0.0.1:11434" in contents


def test_migration_verifies_then_scrubs(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "CTRADER_CLIENT_ID=id",
                "CTRADER_CLIENT_SECRET=secret",
                "CTRADER_ACCESS_TOKEN=token",
                "CTRADER_HOST_TYPE=demo",
                "CTRADER_ACCOUNT_ID=44089601",
                "",
            ]
        ),
        encoding="utf-8",
    )

    vault: dict[str, str] = {}
    monkeypatch.setattr(store, "_is_windows", lambda: True)
    monkeypatch.setattr(store, "read_windows_credential", lambda name: vault.get(name))
    monkeypatch.setattr(
        store,
        "write_windows_credential",
        lambda name, value: vault.__setitem__(name, value),
    )
    monkeypatch.setattr(
        store,
        "delete_windows_credential",
        lambda name: vault.pop(name, None) is not None,
    )

    store.migrate_ctrader_secrets_from_env_file(env_path, scrub_env=True)

    assert vault == {
        "CTRADER_CLIENT_ID": "id",
        "CTRADER_CLIENT_SECRET": "secret",
        "CTRADER_ACCESS_TOKEN": "token",
    }
    contents = env_path.read_text(encoding="utf-8")
    assert "CTRADER_CLIENT_SECRET=" not in contents
    assert "CTRADER_ACCESS_TOKEN=" not in contents
    assert "CTRADER_HOST_TYPE=demo" in contents


def test_failed_migration_restores_previous_vault_values(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "CTRADER_CLIENT_ID=new-id",
                "CTRADER_CLIENT_SECRET=new-secret",
                "CTRADER_ACCESS_TOKEN=new-token",
                "",
            ]
        ),
        encoding="utf-8",
    )

    vault = {
        "CTRADER_CLIENT_ID": "old-id",
        "CTRADER_CLIENT_SECRET": "old-secret",
        "CTRADER_ACCESS_TOKEN": "old-token",
    }
    monkeypatch.setattr(store, "_is_windows", lambda: True)
    monkeypatch.setattr(store, "read_windows_credential", lambda name: vault.get(name))

    writes = 0

    def write(name: str, value: str) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise store.CredentialStoreError("simulated write failure")
        vault[name] = value

    monkeypatch.setattr(store, "write_windows_credential", write)
    monkeypatch.setattr(
        store,
        "delete_windows_credential",
        lambda name: vault.pop(name, None) is not None,
    )

    with pytest.raises(store.CredentialStoreError, match="simulated write failure"):
        store.migrate_ctrader_secrets_from_env_file(env_path, scrub_env=True)

    assert vault == {
        "CTRADER_CLIENT_ID": "old-id",
        "CTRADER_CLIENT_SECRET": "old-secret",
        "CTRADER_ACCESS_TOKEN": "old-token",
    }
    assert "CTRADER_ACCESS_TOKEN=new-token" in env_path.read_text(encoding="utf-8")
