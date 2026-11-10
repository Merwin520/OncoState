"""Experiment 2.2 data prep: one BL/FU CT pair per development patient (series 00).

Settings (same downstream pipeline; only the lesion source differs):
  S0  GT lesion masks, ids shuffled (no Find, no Cut)         -> isolates Match/Status/Measure
  A   GT lesion centroids -> ULS+ Cut                         -> adds Cut error
  B   nnU-Net Find components -> ULS+ Cut                     -> adds Find error (fully automatic)
Candidate ids are re-drawn as shuffled sequential integers so identity can never leak through
label values; GT only supplies centroids (A) or shapes (S0) and is otherwise evaluation-only.
"""
import argparse, json, os, random
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('EXP22_OUT', ROOT / 'outputs/exp22_20260918'))
DATA = Path(os.path.join(os.environ.get('ONCOSTATE_DATA', 'data'), 'Longitudinal-CT'))
TARGET = '006f52e910'


def pairs():
    m = json.loads((ROOT / 'configs/development_manifest.json').read_text())
    if os.environ.get('EXP22_ALL'):
        # radiologist-position protocol on the whole dataset (no trained model involved): every subject with series-00
        # BL/FU CT and both GT masks. cohort tag: dev = development manifest + single-patient target, locked = everyone else.
        dev = set(m['subject_ids']) | {TARGET}
        out = []
        for bl in sorted((DATA / 'inputsTr').glob('*_BL_img_BL_img_00.nii.gz')):
            pid = bl.name.split('_BL_img_')[0]
            fu = DATA / 'inputsTr' / f'{pid}_FU_img_FU_img_00.nii.gz'
            if fu.is_file() and gt_path(pid, 'baseline') and gt_path(pid, 'followup'):
                out.append({'patient_id': pid, 'baseline_ct': str(bl), 'followup_ct': str(fu), 'cohort': 'dev' if pid in dev else 'locked'})
        return out
    if TARGET in m['subject_ids']:
        raise RuntimeError('target in dev')
    by = {}
    for s in m['scans']:
        by.setdefault(s['subject_id'], {})[(s['timepoint'], s['series'])] = s['ct_path']
    out = []
    for pid in sorted(by):
        if ('baseline', '00') in by[pid] and ('followup', '00') in by[pid]:
            out.append({'patient_id': pid, 'baseline_ct': by[pid][('baseline', '00')],
                        'followup_ct': by[pid][('followup', '00')]})
    return out


def gt_path(pid, tp):
    pre, folder = ('BL', 'inputsTr') if tp == 'baseline' else ('FU', 'targetsTr')
    p = DATA / folder / f'{pid}_{pre}_mask_{pre}_img_00.nii.gz'
    return p if p.is_file() else None


def save_labels(arr, ref, path):
    img = sitk.GetImageFromArray(arr.astype(np.uint16))
    img.CopyInformation(ref)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['gt', 'find'], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plist = pairs()
    (OUT / 'pairs.json').write_text(json.dumps(plist, indent=1))
    sh, nsh = (int(x) for x in os.environ.get('EXP22_SHARD', '0/1').split('/'))
    plist = plist[sh::nsh]
    jobs_file = OUT / ('cut_jobs.json' if nsh == 1 else f'cut_jobs_shard{sh}.json')
    jobs = {}
    if jobs_file.exists():
        jobs = json.loads(jobs_file.read_text())
    for p in plist:
        pid = p['patient_id']
        for tp, ct in (('baseline', p['baseline_ct']), ('followup', p['followup_ct'])):
            rd = sitk.ImageFileReader(); rd.SetFileName(ct); rd.ReadImageInformation()
            ref = sitk.Image(rd.GetSize(), sitk.sitkUInt8)
            ref.SetSpacing(rd.GetSpacing()); ref.SetOrigin(rd.GetOrigin()); ref.SetDirection(rd.GetDirection())
            if a.stage == 'gt':
                gp = gt_path(pid, tp)
                if gp is None:
                    continue
                if (OUT / f'gtcand/{pid}_{tp}.nii.gz').exists() and (OUT / f'gtcand/{pid}_{tp}.evalmap.json').exists():
                    jobs[f'A:{pid}:{tp}'] = {'setting': 'A', 'ct': ct, 'cand': str(OUT / f'gtcand/{pid}_{tp}.nii.gz'),
                                             'out': str(OUT / f'cut_A/{pid}_{tp}.nii.gz')}
                    continue
                gimg = sitk.ReadImage(str(gp))
                if not (gimg.GetSize() == ref.GetSize() and np.allclose(gimg.GetSpacing(), ref.GetSpacing())):
                    print('SKIP (GT/CT geometry mismatch):', pid, tp, flush=True)
                    continue
                g = sitk.GetArrayFromImage(gimg)
                ids = [int(i) for i in np.unique(g) if i]
                rng = random.Random(f'exp22:{pid}:{tp}')
                new = list(range(1, len(ids) + 1)); rng.shuffle(new)
                relab = np.zeros(g.shape, np.uint16)
                emap = {}
                for gid, nid in zip(ids, new):
                    relab[g == gid] = nid; emap[nid] = gid
                save_labels(relab, ref, OUT / f'gtcand/{pid}_{tp}.nii.gz')
                (OUT / f'gtcand/{pid}_{tp}.evalmap.json').write_text(json.dumps(emap))
                jobs[f'A:{pid}:{tp}'] = {'setting': 'A', 'ct': ct, 'cand': str(OUT / f'gtcand/{pid}_{tp}.nii.gz'),
                                         'out': str(OUT / f'cut_A/{pid}_{tp}.nii.gz')}
            else:
                pred = ROOT / f'outputs/find_nnunet_dev_20260918/pred/{pid}_{"BL" if tp == "baseline" else "FU"}_00.nii.gz'
                if not pred.is_file():
                    print('SKIP (Find prediction not ready):', pred.name, flush=True)
                    continue
                lab, n = ndimage.label(sitk.GetArrayFromImage(sitk.ReadImage(str(pred))) > 0)
                save_labels(lab, ref, OUT / f'findcand/{pid}_{tp}.nii.gz')
                jobs[f'B:{pid}:{tp}'] = {'setting': 'B', 'ct': ct, 'cand': str(OUT / f'findcand/{pid}_{tp}.nii.gz'),
                                         'out': str(OUT / f'cut_B/{pid}_{tp}.nii.gz'), 'n_candidates': int(n)}
        print(pid, a.stage, flush=True)
    jobs_file.write_text(json.dumps(jobs, indent=1))
    print(len(jobs), 'cut jobs total')


if __name__ == '__main__':
    main()
