"""Automated sanity tests for the state-semantics ablation.  Run after run_all.sh:  python -m pytest -q test_ablation.py"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanity  # noqa: E402
import variants as V  # noqa: E402
from features.measurement_features import ISPY2EvidenceSource, ispy2_cfg  # noqa: E402


@pytest.fixture(scope='module')
def src():
    return ISPY2EvidenceSource()


@pytest.mark.parametrize('check', sanity.CHECKS, ids=[c.__name__ for c in sanity.CHECKS])
def test_sanity(check):
    r = check()
    assert r['passed'], r['detail']


def test_strip_provenance_keeps_values(src):
    pid = sorted(src.visits.patient_id.unique())[0]; cfg = ispy2_cfg(pid)
    s = None
    for tp in cfg['timepoints']:
        s, _ = V.synchronize(s, tp, src.observation(pid, tp), clinical=src.clinical(pid) if s is None else None, cfg=cfg)
    e = V.strip_provenance(s)
    assert V.values_only(e) == V.values_only(s) and e['tumour_state'] == s['tumour_state']
    assert not any(x['provenance_complete'] for x in V.provenance_audit(e))
    assert all(x['provenance_complete'] for x in V.provenance_audit(s))


def test_mutable_store_keeps_only_current(tmp_path, src):
    pid = sorted(src.visits.patient_id.unique())[0]; cfg = ispy2_cfg(pid); st = V.MutableStore(tmp_path)
    s = None
    for tp in cfg['timepoints']:
        s, ev = V.synchronize(s, tp, src.observation(pid, tp), clinical=src.clinical(pid) if s is None else None, cfg=cfg)
        st.write(s, ev)
    assert [p.name for p in (tmp_path / pid).iterdir()] == ['current.json']
    with pytest.raises(V.NotRecoverable):
        st.load(pid, 'v0')
    assert V.values_only(st.latest(pid)) == V.values_only(s)


def test_implicit_sync_records_nothing_for_absent_visit(src):
    pid = sorted(src.visits.patient_id.unique())[0]; cfg = ispy2_cfg(pid)
    s, ev = V.synchronize_implicit(None, 'T0', src.observation(pid, 'T0'), clinical=src.clinical(pid), cfg=cfg)
    s2, ev2 = V.synchronize_implicit(s, 'T1', {'status': 'hidden_stress_test'}, cfg=cfg)
    assert ev2 is None and s2 is s and 'T1' not in s2['history'] and not s2['observation_status']
    s3, _ = V.synchronize_implicit(s2, 'T2', src.observation(pid, 'T2'), cfg=cfg)
    s4, _ = V.synchronize_implicit(s3, 'T1', src.observation(pid, 'T1'), cfg=cfg, event_type='late_arrival')
    assert s4['history']['T1']['status'] == 'observed' and s4['current_timepoint'] == 'T2'   # never moves backwards
    with pytest.raises(V.HistoricalWriteRefused):
        V.synchronize_implicit(s4, 'T1', src.observation(pid, 'T1'), cfg=cfg)


def test_official_code_untouched():
    """The ablation imports the official functions; it must not monkey-patch them."""
    import state.synchronization as S
    import state.versioning as VS
    assert V.synchronize is S.synchronize and V.revise is S.revise and V.VersionStore is VS.VersionStore
    assert VS.VersionStore.write is not V.ProvenanceFreeStore.write
