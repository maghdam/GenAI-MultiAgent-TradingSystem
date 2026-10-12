from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.security.credential_store import (  # noqa: E402
    ALPACA_SECRET_ENV_NAMES,
    CredentialStoreError,
    alpaca_credential_status,
    delete_windows_alpaca_credential,
    store_alpaca_secrets,
)


def _print_status() -> None:
    status = alpaca_credential_status()
    for name in ALPACA_SECRET_ENV_NAMES:
        row = status[name]
        print(
            f"{name}: "
            f"windows_credential_manager={'present' if row['windows_credential_manager'] else 'absent'}, "
            f"environment={'present' if row['environment'] else 'absent'}, "
            f"effective_source={row['effective_source']}"
        )


def _set_interactive() -> None:
    entered: dict[str, str] = {}
    for name in ALPACA_SECRET_ENV_NAMES:
        value = getpass.getpass(f"{name}: ").strip()
        if not value:
            raise CredentialStoreError(f"{name} must not be empty.")
        entered[name] = value

    store_alpaca_secrets(entered)
    print("Stored and verified Alpaca Paper credentials in Windows Credential Manager.")


def _delete_all() -> None:
    deleted = 0
    for name in ALPACA_SECRET_ENV_NAMES:
        if delete_windows_alpaca_credential(name):
            deleted += 1
    suffix = "entry" if deleted == 1 else "entries"
    print(f"Deleted {deleted} TradeAgent Alpaca credential {suffix}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage TradeAgent Alpaca Paper API credentials in Windows Credential "
            "Manager without printing secret values."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "status",
        help="Show presence/source metadata only; never print credential values.",
    )
    subparsers.add_parser(
        "set",
        help="Prompt securely for the Alpaca Paper API key ID and secret key.",
    )
    subparsers.add_parser(
        "delete",
        help="Delete TradeAgent Alpaca credential entries from Windows Credential Manager.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.command == "status":
            _print_status()
        elif args.command == "set":
            _set_interactive()
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
