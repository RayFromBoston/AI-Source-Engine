"""Run AL-1.0 decode-step benchmark smoke test."""

from al10.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["bench-smoke", "--steps", "1000", "--heads", "32", "--key-len", "1024"]))
