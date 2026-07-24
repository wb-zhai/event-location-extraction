"""Create Argilla users and assign them to workspaces.

See scripts/annotations/README.md for setup and usage.
"""

import argparse
import os
import sys
from enum import Enum
from pathlib import Path

import argilla as rg

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv


class Role(str, Enum):
    ANNOTATOR = "annotator"
    ADMIN = "admin"
    OWNER = "owner"


def get_client():

    api_url = os.environ.get("ARGILLA_API_URL")
    api_key = os.environ.get("ARGILLA_API_KEY")
    if not api_url or not api_key:
        raise SystemExit(
            "ARGILLA_API_URL and ARGILLA_API_KEY must be set (see .env or "
            "scripts/annotations/README.md)."
        )
    return rg.Argilla(api_url=api_url, api_key=api_key)


def get_or_create_workspace(client, name: str):

    workspace = client.workspaces(name)
    if workspace is not None:
        return workspace
    workspace = rg.Workspace(name=name, client=client)
    workspace.create()
    print(f"Created workspace {name!r}.")
    return workspace


def add_user(args: argparse.Namespace) -> None:
    client = get_client()

    user = client.users(args.username)
    if user is not None:
        print(f"User {args.username!r} already exists, skipping creation.")
    else:
        if not args.password:
            raise SystemExit("--password is required when creating a new user.")
        user = rg.User(
            username=args.username,
            password=args.password,
            first_name=args.first_name or args.username,
            last_name=args.last_name,
            role=args.role.value,
            client=client,
        )
        user.create()
        print(f"Created user {args.username!r} with role {args.role.value!r}.")

    for workspace_name in args.workspace:
        if args.create_workspace:
            workspace = get_or_create_workspace(client, workspace_name)
        else:
            workspace = client.workspaces(workspace_name)
            if workspace is None:
                raise SystemExit(
                    f"Workspace {workspace_name!r} not found. "
                    "Pass --create-workspace to create it automatically."
                )
        user.add_to_workspace(workspace)
        print(f"Added {args.username!r} to workspace {workspace_name!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Argilla users.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to a .env file with ARGILLA_API_URL / ARGILLA_API_KEY (default: repo .env).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Create a new Argilla user.")
    add_parser.add_argument("--username", required=True, type=str)
    add_parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Required when creating a new user; ignored if the user already exists.",
    )
    add_parser.add_argument("--first-name", type=str, default=None)
    add_parser.add_argument("--last-name", type=str, default=None)
    add_parser.add_argument(
        "--role",
        type=Role,
        choices=list(Role),
        default=Role.ANNOTATOR,
        metavar="{" + ",".join(r.value for r in Role) + "}",
        help="One of: annotator, admin, owner. Only 'owner' can create workspaces.",
    )
    add_parser.add_argument(
        "--workspace",
        type=str,
        action="append",
        default=[],
        help="Workspace to add the user to. Repeat for multiple workspaces.",
    )
    add_parser.add_argument(
        "--create-workspace",
        action="store_true",
        help="Create any --workspace that doesn't already exist (requires owner role).",
    )
    add_parser.set_defaults(func=add_user)

    args = parser.parse_args()
    load_dotenv(args.env_file)
    args.func(args)


if __name__ == "__main__":
    main()
