"""Compatibility launcher for the installed whfm sweep runner."""

from whfm.sweep import main


if __name__ == "__main__":
    raise SystemExit(main())
