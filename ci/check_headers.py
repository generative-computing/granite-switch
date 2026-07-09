# SPDX-License-Identifier: Apache-2.0

"""Check that source files have required SPDX headers."""

import argparse
import sys
from pathlib import Path

HEADER_LINES = [
    "# SPDX-License-Identifier: Apache-2.0",
]


def has_headers(content: str) -> bool:
    """Check if content has required headers in first 5 lines."""
    first_lines = "\n".join(content.split("\n")[:5])
    return all(header in first_lines for header in HEADER_LINES)


def add_headers(filepath: Path) -> bool:
    """Add headers to file if missing. Returns True if file was modified."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        if has_headers(content):
            return False

        lines = content.split("\n")
        if lines and lines[0].startswith("#!"):
            new_content = "\n".join([lines[0]] + HEADER_LINES + [""] + lines[1:])
        else:
            new_content = "\n".join([*HEADER_LINES, "", content])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def check_file(filepath: Path) -> bool:
    """Check if file has required headers."""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        return has_headers(content)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def main() -> int:
    """Check or fix SPDX headers in files."""
    parser = argparse.ArgumentParser(
        description="Check/fix SPDX headers in source files"
    )
    parser.add_argument("files", nargs="+", type=Path, help="Files to check")
    parser.add_argument(
        "--fix", action="store_true", help="Automatically add missing headers"
    )
    args = parser.parse_args()

    if args.fix:
        modified = []
        for filepath in args.files:
            if add_headers(filepath):
                modified.append(filepath)

        if modified:
            print("Added SPDX headers to:")
            for f in modified:
                print(f"  - {f}")
            return 1  # Return 1 so pre-commit knows files were modified

        return 0

    else:
        missing_headers = []
        for filepath in args.files:
            if not check_file(filepath):
                missing_headers.append(filepath)

        if missing_headers:
            print("❌ Error: The following files are missing required SPDX headers:")
            for f in missing_headers:
                print(f"  - {f}")
            print("\nRequired header (at top of file):")
            for header in HEADER_LINES:
                print(f"  {header}")
            return 1

        print("✅ All files have required SPDX headers")
        return 0


if __name__ == "__main__":
    sys.exit(main())
