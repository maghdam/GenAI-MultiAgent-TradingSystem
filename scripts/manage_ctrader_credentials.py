from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.security.credential_store import (  # noqa: E402
    CTRADER_SECRET_ENV_NAMES,
    CredentialStoreError,
    ctrader_credential_status,
    delete_windows_credential,
    migrate_ctrader_secrets_from_env_file,
    read_windows_credential,
    restore_ctrader_secrets_to_env_file,
    write_windows_credential,
)


DEFAULT_ENV_FILE = REPO_ROOT / "backend" / ".env"


def _env_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _print_status(path: Path) -> None:
    status = ctrader_credential_status(path)
    print(f"Environment file: {path}")
    for name in CTRADER_SECRET_ENV_NAMES:
        row = status[name]
        print(
            f"{name}: "
            f"windows_credential_manager={'present' if row['windows_credential_manager'] else 'absent'}, "
            f"env_file={'present' if row['env_file'] else 'absent'}, "
            f"effective_source={row['effective_source']}"
        )


def _set_interactive() -> None:
    entered: dict[str, str] = {}
    for name in CTRADER_SECRET_ENV_NAMES:
        value = getpass.getpass(f"{name}: ").strip()
        if not value:
            raise CredentialStoreError(f"{name} must not be empty.")
        entered[name] = value

    for name, value in entered.items():
        write_windows_credential(name, value)

    failed = [
        name
        for name, value in entered.items()
        if read_windows_credential(name) != value
    ]
    if failed:
        raise CredentialStoreError(
            "Credential write verification failed for: " + ", ".join(failed)
        )
    print("Stored and verified all cTrader credentials in Windows Credential Manager.")


def _delete_all() -> None:
    deleted = 0
    for name in CTRADER_SECRET_ENV_NAMES:
        if delete_windows_credential(name):
            deleted += 1
    suffix = "entry" if deleted == 1 else "entries"
    print(f"Deleted {deleted} TradeAgent cTrader credential {suffix}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage TradeAgent cTrader secrets in Windows Credential Manager "
            "without printing secret values."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser(
        "status",
        help="Show presence/source metadata only; never print credential values.",
    )
    status_parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the local .env file used for fallback/bootstrap configuration.",
    )

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Copy cTrader secrets from .env into Windows Credential Manager.",
    )
    migrate_parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Source .env file.",
    )
    migrate_parser.add_argument(
        "--scrub-env",
        action="store_true",
        help=(
            "After verified Credential Manager writes, remove the three cTrader "
            "secret assignments from the .env file."
        ),
    )

    subparsers.add_parser(
        "set",
        help="Prompt securely for the three cTrader credential values and store them.",
    )

    restore_parser = subparsers.add_parser(
        "restore-env",
        help=(
            "Emergency rollback: restore cTrader secret assignments from Windows "
            "Credential Manager into the local .env file."
        ),
    )
    restore_parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Destination .env file.",
    )

    subparsers.add_parser(
        "delete",
        help="Delete TradeAgent cTrader credential entries from Windows Credential Manager.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.command == "status":
            _print_status(_env_path(args.env_file))
        elif args.command == "migrate":
            env_path = _env_path(args.env_file)
            migrate_ctrader_secrets_from_env_file(
                env_path,
                scrub_env=bool(args.scrub_env),
            )
            action = (
                "migrated, verified, and scrubbed from .env"
                if args.scrub_env
                else "migrated and verified"
            )
            print(f"cTrader credentials {action}.")
            _print_status(env_path)
        elif args.command == "set":
            _set_interactive()
        elif args.command == "restore-env":
            env_path = _env_path(args.env_file)
            restore_ctrader_secrets_to_env_file(env_path)
            print(f"Restored cTrader secret assignments to {env_path}.")
            _print_status(env_path)
        elif args.command == "delete":
            _delete_all()
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (CredentialStoreError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
