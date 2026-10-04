# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Command line of the machine tuner: ``python -m ventorum.hardware`` or ``ventorum-tune``."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    """Run the command line. Return the exit code."""
    parser = argparse.ArgumentParser(prog="ventorum-tune", description="Tune Ventorum for this machine.")
    parser.add_argument("command", nargs="?", default="tune", choices=("tune", "show", "path"),
                        help="tune (default): measure and save the profile; show: print the profile; path: print its file")
    parser.add_argument("--quick", action="store_true", help="skip the large cases")
    args = parser.parse_args(argv)

    from ventorum.hardware import active_profile, profile_path, read_profile, tune_machine

    if args.command == "path":
        print(profile_path())
        return 0
    if args.command == "show":
        data = read_profile()
        if data is None:
            print("No profile. Run: ventorum-tune")
            return 1
        print(json.dumps(data, indent=2, sort_keys=True))
        print("Matches this machine:", active_profile() is not None)
        return 0
    tune_machine(quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
