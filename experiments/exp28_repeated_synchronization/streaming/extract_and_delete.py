"""Validated-only raw-data deletion helper and compatibility entry point."""
import shutil
from pathlib import Path

def delete_raw(path, validated=False, diagnostics_recorded=False):
    """Delete raw imaging only once its outcome is on record: validated extraction, or a recorded failure diagnosis."""
    if not (validated or diagnostics_recorded):
        raise ValueError('raw data may be deleted only after validation or after the failure diagnosis is recorded')
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    return not target.exists()


def process_eligible(*args, **kwargs):
    from streaming.processor import process_eligible as run
    return run(*args, **kwargs)


__all__ = ['process_eligible', 'delete_raw']
