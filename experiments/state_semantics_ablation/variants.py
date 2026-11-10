"""Single-mechanism ablations of the official OncoState state semantics (Exp 2.8 I-SPY2 implementation and Exp 2.2 CT state).

Every variant reuses the official code for everything except the one mechanism it removes:
  A  Full                 official synchronize() + VersionStore (unchanged; imported, not copied)
  B  Mutable state        same synchronize() and schema; storage overwrites one current record in place (no write-once versions,
                          no parent chain, no SHA-256 lineage, version metadata cleared)
  D  Implicit missingness same versioned storage and provenance; an expected-but-absent visit leaves no state record (no
                          observation_status / missingness entry, no version), and delayed evidence is merged with a conventional
                          upsert (stored as 'observed', current_timepoint = most recently updated visit), no typed late-arrival transition
  E  No provenance        official synchronize() + VersionStore; field-level provenance, the provenance log, clinical provenance and
                          source-observation identifiers are removed before storage; every value is kept
  F  Non-durable correction  the verified correction is applied to the result returned at review time only; it is not written to the
                          stored state, so later queries/synchronizations start from the uncorrected state
  C  (no persistent lesion identity) is NOT implemented: see ablation_config.yaml (not separable in the existing two-timepoint CT state).
"""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'experiments/exp28_repeated_synchronization'))
from state.patient_state import derive_tumour_state, new_state  # noqa: E402
from state.provenance import complete  # noqa: E402
from state.synchronization import HistoricalWriteRefused, observation_paths, revise, synchronize  # noqa: E402,F401
from state.versioning import VersionStore  # noqa: E402

VARIANTS = ['A_full', 'B_mutable', 'D_implicit_missingness', 'E_no_provenance', 'F_nondurable_correction']


# ------------------------------------------------------------------------------------------------- B: mutable storage
class NotRecoverable(RuntimeError):
    pass


class MutableStore:
    """Conventional mutable patient record: patient/<id>/current.json overwritten on every update. No versions, parent pointers or hashes."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def _f(self, patient_id):
        d = self.root / str(patient_id); d.mkdir(parents=True, exist_ok=True)
        return d / 'current.json'

    def write(self, state, event=None):
        s = copy.deepcopy(state)
        s['state_version'] = None; s['parent_version'] = None          # version-lineage metadata disabled
        self._f(state['patient_id']).write_text(json.dumps(s, indent=1, sort_keys=True, default=str))

    def latest(self, patient_id):
        f = self._f(patient_id)
        return json.loads(f.read_text()) if f.exists() else None

    def load(self, patient_id, state_version):
        raise NotRecoverable('mutable record keeps only the current state')

    def all_versions(self, patient_id):
        s = self.latest(patient_id)
        return [s] if s else []

    def verify(self, patient_id):
        return None                                                    # no integrity mechanism exists

    def lineage(self, patient_id):
        return {'patient_id': str(patient_id), 'versions': []}


# ------------------------------------------------------------------------------------------------- D: implicit missingness
def synchronize_implicit(prev, timepoint, observation, *, clinical=None, cfg, event_type='scheduled', now=None):
    """Conventional update without typed missingness or late-arrival semantics.

    Observed evidence is upserted into the same history structure (same values, same field provenance, same versioned storage);
    absent evidence changes nothing (returns (prev, None): no record that an expected visit is missing, no version). current_timepoint
    is the latest observed visit in protocol order (a reasonable conventional choice: delayed evidence never moves it backwards). The write guard on existing observations is kept (it belongs to the transition operator,
    not to the missingness semantics)."""
    if prev is None:
        prev = new_state(cfg['patient_id'], cfg['timepoints'], clinical, cfg['phase_names'])
        prev['state_version'] = None
    if observation is None or observation.get('status') != 'observed':
        return prev, None
    if timepoint in prev['history']:
        raise HistoricalWriteRefused(f'{timepoint} already in history')
    s = copy.deepcopy(prev)
    s['version_index'] = prev['version_index'] + 1
    s['state_version'] = f"v{s['version_index']}"
    s['parent_version'] = prev['state_version']
    s['sync_event_id'] = f"{s['patient_id']}:{s['state_version']}:{timepoint}:update"
    s['updated_at'] = now
    rec = copy.deepcopy(observation); rec['status'] = 'observed'; rec['synchronized_in'] = s['state_version']
    s['history'][timepoint] = rec
    order = s['protocol_timepoints']
    s['current_timepoint'] = max(s['history'], key=order.index)
    s['treatment_context']['phase_at_current'] = cfg['phase_names'].get(s['current_timepoint'])
    s['source_observation'] = dict(timepoint=timepoint, study_uid=rec.get('study_uid'), series_uids=rec.get('series_uids', []))
    s['tumour_state'] = derive_tumour_state(s, cfg['measure_keys'])
    s['QC_flags'] = sorted(set(s['QC_flags']) | set(rec.get('qc_flags', [])))
    s['provenance'].append(dict(sync_event_id=s['sync_event_id'], timepoint=timepoint, state_version=s['state_version'],
                                source_study_uid=rec.get('study_uid'), source_series_uids=rec.get('series_uids', []), at=now))
    s['events'].append(s['sync_event_id'])
    return s, dict(sync_event_id=s['sync_event_id'], patient_id=s['patient_id'], timepoint=timepoint, event_type='update')


# ------------------------------------------------------------------------------------------------- E: no provenance
def strip_provenance(state):
    """Remove field-level provenance, the provenance log, clinical provenance and source identifiers. Values are untouched."""
    s = copy.deepcopy(state)
    for rec in s.get('history', {}).values():
        for f in rec.get('fields', {}).values():
            f['provenance'] = None
        rec.pop('study_uid', None); rec.pop('series_uids', None)
    s['provenance'] = []
    s['clinical_provenance'] = None
    for r in s.get('revisions', []):
        r['source'] = None; r['actor'] = None
    if s.get('source_observation'):
        s['source_observation'] = {k: v for k, v in s['source_observation'].items() if k not in ('study_uid', 'series_uids')}
    return s


def revise_without_provenance(prev, timepoint, field_name, new_value, *, reason, actor, source, cfg=None, now=None):
    """Official revise() semantics (new version, revision log with old/new value) without provenance: the revision source/actor and
    field provenance are not recorded."""
    s = copy.deepcopy(prev)
    old = s['history'][timepoint]['fields'][field_name]
    s['version_index'] += 1
    s['state_version'] = f"v{s['version_index']}"
    s['parent_version'] = prev['state_version']
    s['sync_event_id'] = f"{s['patient_id']}:{s['state_version']}:{timepoint}:revision"
    s['updated_at'] = now
    rev = dict(revision_id=s['sync_event_id'], timepoint=timepoint, field=field_name, old_value=old['value'], new_value=new_value,
               reason=reason, actor=None, source=None, state_version=s['state_version'], at=now)
    s['history'][timepoint]['fields'][field_name] = dict(old, value=new_value, revised_by=rev['revision_id'], provenance=None)
    s['revisions'].append(rev)
    s['events'].append(s['sync_event_id'])
    if cfg is not None:
        s['tumour_state'] = derive_tumour_state(s, cfg['measure_keys'])
    return s, rev


class ProvenanceFreeStore(VersionStore):
    """Official write-once versioned store; provenance is stripped from every state before it is written."""
    def write(self, state, event=None):
        return super().write(strip_provenance(state), event)


def values_only(state):
    """Clinical content without provenance / identifiers (used to check that E keeps every value)."""
    return {tp: {k: f.get('value') for k, f in rec.get('fields', {}).items()} for tp, rec in state.get('history', {}).items()}


def provenance_audit(state):
    """Per observed field: provenance complete, source study/series recoverable, source class identifiable."""
    rows = []
    for tp, rec in state.get('history', {}).items():
        for k, f in rec.get('fields', {}).items():
            if f.get('value') is None:
                continue
            p = f.get('provenance')
            rows.append(dict(timepoint=tp, field=k, provenance_complete=bool(complete(p)),
                             source_recoverable=bool(p and (p.get('source_study_uid') or p.get('source_file'))),
                             source_class_identifiable=bool(p and p.get('source_type'))))
    return rows
