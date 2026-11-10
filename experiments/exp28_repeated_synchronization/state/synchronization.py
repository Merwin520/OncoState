"""Synchronization of one persistent patient state with arriving observations, and the stateless comparator.

synchronize(prev, ...) is the ONLY state-transition function. Both strategies use it:
  persistent  : load v(k-1) from the version store, read only the new observation (T_k), apply synchronize once, write v_k;
  stateless   : re-read clinical record + observations T0..T_k from the source, apply synchronize k+1 times from scratch.
Every source access goes through an EvidenceSource, which counts reads, series reads and tool calls.
"""
import copy
import time

from state.patient_state import derive_tumour_state, new_state
from state.schemas import PROCESSING_VERSION


class HistoricalWriteRefused(RuntimeError):
    """An observation for this timepoint already exists; only an explicit revision may change it."""


def _flatten(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(d, list):
        out[prefix] = repr(d)
    else:
        out[prefix] = d
    return out


def observation_paths(state):
    """Flattened historical observation content (provenance included) — the object of the invariance check."""
    return _flatten({'history': state.get('history', {}), 'clinical_context': state.get('clinical_context', {})})


def synchronize(prev, timepoint, observation, *, clinical=None, cfg, event_type='scheduled', counter=None, now=None):
    """Return (new_state, event). observation is None when no study exists for this timepoint in the source."""
    t0 = time.perf_counter()
    if prev is None:
        if clinical is None:
            raise ValueError('the first synchronization needs the clinical record')
        prev = new_state(cfg['patient_id'], cfg['timepoints'], clinical, cfg['phase_names'])
    if timepoint not in prev['protocol_timepoints']:
        raise ValueError(f'unknown timepoint {timepoint}')
    s = copy.deepcopy(prev)
    s['version_index'] = prev['version_index'] + 1
    s['state_version'] = f"v{s['version_index']}"
    s['parent_version'] = prev['state_version']
    s['sync_event_id'] = f"{s['patient_id']}:{s['state_version']}:{timepoint}:{event_type}"
    s['updated_at'] = now
    order = prev['protocol_timepoints']
    if event_type == 'late_arrival':
        if prev['observation_status'].get(timepoint) == 'observed':
            raise HistoricalWriteRefused(f'{timepoint} already observed')
    elif order.index(timepoint) < order.index(prev['current_timepoint']) if prev['current_timepoint'] else False:
        raise ValueError('scheduled synchronization must follow protocol order; use event_type="late_arrival" for delayed evidence')
    if timepoint in prev['history'] and prev['history'][timepoint]['status'] in ('observed', 'late_arrival'):
        raise HistoricalWriteRefused(f'{timepoint} already in history')
    if event_type != 'late_arrival':
        s['current_timepoint'] = timepoint
        s['treatment_context']['phase_at_current'] = cfg['phase_names'].get(timepoint)
    if observation is not None and observation.get('status') == 'observed':
        rec = copy.deepcopy(observation)
        rec['status'] = 'late_arrival' if event_type == 'late_arrival' else 'observed'
        rec['synchronized_in'] = s['state_version']
        s['history'][timepoint] = rec
        s['source_observation'] = dict(timepoint=timepoint, study_uid=rec.get('study_uid'),
                                       series_uids=rec.get('series_uids', []), event_type=event_type)
        s['observation_status'][timepoint] = rec['status']
        s['missingness']['visits'].pop(timepoint, None)
        for k, f in rec['fields'].items():
            if f['value'] is None:
                s['missingness']['fields'].setdefault(timepoint, {})[k] = f.get('missing_reason', 'not available in source')
    else:
        status = 'hidden_stress_test' if (observation or {}).get('status') == 'hidden_stress_test' else 'not_observed'
        s['observation_status'][timepoint] = status
        s['missingness']['visits'][timepoint] = dict(status=status, recorded_in=s['state_version'],
                                                     reason=(observation or {}).get('reason', 'no study for this timepoint in the source at synchronization time'))
        s['source_observation'] = dict(timepoint=timepoint, study_uid=None, series_uids=[], event_type=event_type,
                                       status=status)
    s['tumour_state'] = derive_tumour_state(s, cfg['measure_keys'])
    new_obs = s['history'].get(timepoint) if s['observation_status'].get(timepoint) in ('observed', 'late_arrival') else None
    s['QC_flags'] = sorted(set(s['QC_flags']) | set((new_obs or {}).get('qc_flags', [])))
    s['provenance'].append(dict(sync_event_id=s['sync_event_id'], timepoint=timepoint, event_type=event_type, state_version=s['state_version'],
                                source_study_uid=(new_obs or {}).get('study_uid'), source_series_uids=(new_obs or {}).get('series_uids', []),
                                processing_version=PROCESSING_VERSION, at=now))
    s['events'].append(s['sync_event_id'])
    a, b = observation_paths(prev), observation_paths(s)
    event = dict(sync_event_id=s['sync_event_id'], patient_id=s['patient_id'], previous_state_version=prev['state_version'],
                 new_state_version=s['state_version'], timepoint=timepoint, event_type=event_type,
                 source_study_uid=(new_obs or {}).get('study_uid'), source_series_uids=(new_obs or {}).get('series_uids', []),
                 new_fields=sorted(set(b) - set(a)), updated_fields=sorted(k for k in set(a) & set(b) if a[k] != b[k]),
                 unchanged_fields=sorted(k for k in set(a) & set(b) if a[k] == b[k]),
                 removed_fields=sorted(set(a) - set(b)), n_unchanged_fields=len([k for k in set(a) & set(b) if a[k] == b[k]]),
                 missing_fields=sorted(f'{tp}' for tp, v in s['missingness']['visits'].items()) +
                 sorted(f'{tp}.{k}' for tp, d in s['missingness']['fields'].items() for k in d),
                 QC_status='flagged' if (new_obs or {}).get('qc_flags') else 'pass', provenance=s['provenance'][-1],
                 processing_time=time.perf_counter() - t0, **(counter.snapshot() if counter else {}))
    return s, event


def revise(prev, timepoint, field_name, new_value, *, reason, actor, source, cfg=None, now=None):
    """Explicit, logged revision of an existing observation (e.g. a validated correction). The only legal way to change history."""
    s = copy.deepcopy(prev)
    old = s['history'][timepoint]['fields'][field_name]
    s['version_index'] += 1
    s['state_version'] = f"v{s['version_index']}"
    s['parent_version'] = prev['state_version']
    s['sync_event_id'] = f"{s['patient_id']}:{s['state_version']}:{timepoint}:revision"
    s['updated_at'] = now
    rev = dict(revision_id=s['sync_event_id'], timepoint=timepoint, field=field_name, old_value=old['value'], new_value=new_value,
               reason=reason, actor=actor, source=source, state_version=s['state_version'], at=now)
    s['history'][timepoint]['fields'][field_name] = dict(old, value=new_value, revised_by=rev['revision_id'],
                                                         provenance=dict(old['provenance'], revision_source=source))
    s['revisions'].append(rev)
    s['events'].append(s['sync_event_id'])
    if cfg is not None:
        s['tumour_state'] = derive_tumour_state(s, cfg['measure_keys'])
    return s, rev


class Counter:
    def __init__(self):
        self.source_reads = 0
        self.series_reads = 0
        self.tool_calls = 0
        self.state_reads = 0
        self.state_ops = 0

    def snapshot(self):
        return dict(source_reads=self.source_reads, series_reads=self.series_reads, tool_calls=self.tool_calls,
                    state_reads=self.state_reads, state_ops=self.state_ops)


def persistent_step(store, source, patient_id, timepoint, cfg, now=None, event_type='scheduled'):
    """Load the latest version, read ONLY the new timepoint's evidence, synchronize once, write the new version."""
    t0 = time.perf_counter()
    c = Counter()
    source.counter = c
    prev = store.latest(patient_id)
    c.state_reads += prev is not None
    clinical = source.clinical(patient_id) if prev is None else None
    obs = source.observation(patient_id, timepoint)
    s, ev = synchronize(prev, timepoint, obs, clinical=clinical, cfg=cfg, event_type=event_type, counter=None, now=now)
    c.state_ops += 1
    store.write(s, ev)
    ev.update(c.snapshot(), runtime_seconds=time.perf_counter() - t0, strategy='persistent')
    return s, ev


def stateless_rebuild(source, patient_id, upto_timepoint, cfg, now=None):
    """Re-read everything up to T_k from the source and rebuild the representation from scratch (nothing is loaded from a store)."""
    t0 = time.perf_counter()
    c = Counter()
    source.counter = c
    s = None
    clinical = source.clinical(patient_id)
    for tp in cfg['timepoints'][: cfg['timepoints'].index(upto_timepoint) + 1]:
        s, _ = synchronize(s, tp, source.observation(patient_id, tp), clinical=clinical, cfg=cfg, now=now)
        c.state_ops += 1
    return s, dict(c.snapshot(), runtime_seconds=time.perf_counter() - t0, strategy='stateless', patient_id=patient_id, timepoint=upto_timepoint)
