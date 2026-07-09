# SPDX-License-Identifier: Apache-2.0

"""Check if commit message contains DCO sign-off."""

import argparse
import re
import sys
from pathlib import Path


def check_dco(commit_msg_file: Path) -> bool:
    """Check if commit message has DCO sign-off."""
    try:
        with open(commit_msg_file, "r", encoding="utf-8") as f:
            message = f.read()

        pattern = r"^Signed-off-by: .+ <.+@.+>$"
        if re.search(pattern, message, re.MULTILINE):
            return True

        return False

    except Exception as e:
        print(f"Error reading commit message: {e}")
        return False


def main() -> int:
    """Check DCO sign-off in commit message."""
    parser = argparse.ArgumentParser(description="Check DCO sign-off in commit message")
    parser.add_argument("commit_msg_file", type=Path, help="Path to commit message file")
    args = parser.parse_args()

    if not check_dco(args.commit_msg_file):
        print("❌ Error: Commit message is missing DCO sign-off!")
        print("")
        print("Please sign off your commit by using:")
        print("  git commit -s")
        print("")
        print("Or amend your last commit:")
        print("  git commit --amend -s")
        print("")
        print("This adds a 'Signed-off-by' line certifying you agree to the DCO.")
        return 1

    print("✅ Commit has DCO sign-off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
