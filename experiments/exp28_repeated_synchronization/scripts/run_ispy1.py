"""CLI wrapper for the optional I-SPY1 replication."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from external_validation.ispy1 import run  # noqa: E402

if __name__ == '__main__':
    run()
