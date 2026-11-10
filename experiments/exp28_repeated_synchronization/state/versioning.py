"""Write-once version store: patient_states/<patient>/v{k}_<timepoint>.json + lineage.json (with content hashes).

A version file is never overwritten; writing an existing version raises. lineage.json records parent pointers and the
SHA-256 of every file at write time so later tampering or silent rewrites are detectable.
"""
import hashlib
import json
from pathlib import Path


class VersionExists(RuntimeError):
    pass


def _dump(obj):
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


class VersionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.reads = 0
        self.writes = 0

    def pdir(self, patient_id):
        d = self.root / str(patient_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fname(self, state):
        return f"{state['state_version']}_{state['current_timepoint']}.json"

    def write(self, state, event=None):
        d = self.pdir(state['patient_id'])
        f = d / self.fname(state)
        if f.exists():
            raise VersionExists(f'{f.name} already exists for {state["patient_id"]}; versions are immutable')
        raw = _dump(state)
        f.write_text(raw)
        lin = self.lineage(state['patient_id'])
        lin['versions'].append(dict(state_version=state['state_version'], parent_version=state['parent_version'], file=f.name,
                                    timepoint=state['current_timepoint'], sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                    sync_event_id=state.get('sync_event_id'), event_type=(event or {}).get('event_type')))
        (d / 'lineage.json').write_text(_dump(lin))
        self.writes += 1
        return f

    def lineage(self, patient_id):
        f = self.pdir(patient_id) / 'lineage.json'
        return json.loads(f.read_text()) if f.exists() else {'patient_id': str(patient_id), 'versions': []}

    def latest(self, patient_id):
        lin = self.lineage(patient_id)
        if not lin['versions']:
            return None
        self.reads += 1
        return json.loads((self.pdir(patient_id) / lin['versions'][-1]['file']).read_text())

    def load(self, patient_id, state_version):
        for v in self.lineage(patient_id)['versions']:
            if v['state_version'] == state_version:
                self.reads += 1
                return json.loads((self.pdir(patient_id) / v['file']).read_text())
        raise FileNotFoundError(f'{patient_id} {state_version}')

    def all_versions(self, patient_id):
        return [json.loads((self.pdir(patient_id) / v['file']).read_text()) for v in self.lineage(patient_id)['versions']]

    def verify(self, patient_id):
        """Lineage integrity: files exist, hashes unchanged, parents form a chain."""
        lin = self.lineage(patient_id)
        prev = None
        for v in lin['versions']:
            f = self.pdir(patient_id) / v['file']
            if not f.exists() or hashlib.sha256(f.read_text().encode()).hexdigest() != v['sha256']:
                return False
            if v['parent_version'] != prev:
                return False
            prev = v['state_version']
        return bool(lin['versions'])
