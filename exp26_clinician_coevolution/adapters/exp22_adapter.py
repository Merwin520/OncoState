"""Experiment 2.2 -> Experiment 2.6 cohort bundle (real data; runs inside the original OncoState repository).

Reads the frozen Experiment 2.2 outputs WITHOUT modifying them:
  <exp22_dir>/outputs/exp22_baselines/exp22_regen/{pairs.json, state_A/, reg/, cut_A/}
and the Experiment 2.2 reference definitions (exp22b_common.reference), and writes a bundle:
  cases/<pid>.json      lesions B<k>/F<k>, centroid, registered baseline centroid (frozen 2.2 rigid transform),
                        volume, diameter, mask refs, GT-free registration residual
  gt_hidden/<pid>.json  reference pairs / classes / change / burden (offline use only)
  crops/<pid>/...png    optional lesion-centred axial crops for the clinician UI (--crops-for)

Pred-lesion reference classes (gt_hidden):
  baseline_class  persistent | disappearing | partner_undetected | spurious
  followup_class  persistent | newly_appearing | partner_undetected | spurious

Usage (from the repository, environment with SimpleITK; image paths are recorded relative to --image-root):
  python -m adapters.exp22_adapter --exp22-dir ../experiments/exp1_agentic_longitudinal \
         --out data/real/exp22_longitudinal_ct --image-root /path/to/Longitudinal-CT [--crops-for PID ...] [--workers 16]
The real bundle is NOT included in the distributed ZIP.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from adapters.schema_mapping import write_bundle

_EXP22 = None


def _init(exp22_dir: str):
    global _EXP22
    os.environ.setdefault("EXP22_OUT", str(Path(exp22_dir) / "outputs/exp22_baselines/exp22_regen"))
    os.environ.setdefault("EXP22_ALL", "1")
    sys.path.insert(0, str(exp22_dir))
    import exp22b_common as C  # noqa: E402  (repository module, frozen 2.2 definitions)
    _EXP22 = C


def _rel(path: str, root: str | None) -> str:
    if root and str(path).startswith(str(root)):
        return os.path.relpath(path, root)
    return Path(path).name


def _crop(img_path: str, mask_path: str, lesion_label: int, centroid_mm, out_png: Path, fov_mm: float = 96.0, px: int = 192):
    import SimpleITK as sitk
    from PIL import Image
    img, msk = sitk.ReadImage(img_path), sitk.ReadImage(mask_path)
    idx = img.TransformPhysicalPointToIndex(tuple(centroid_mm))
    arr = sitk.GetArrayFromImage(img)[min(max(idx[2], 0), img.GetSize()[2] - 1)]
    lab = sitk.GetArrayFromImage(msk)
    k = min(max(msk.TransformPhysicalPointToIndex(tuple(centroid_mm))[2], 0), msk.GetSize()[2] - 1)
    m = lab[k] == lesion_label
    sp = img.GetSpacing()
    h = int(fov_mm / 2 / sp[0])
    r0, c0 = idx[1], idx[0]
    sl = (slice(max(r0 - h, 0), r0 + h), slice(max(c0 - h, 0), c0 + h))
    win = np.clip((arr[sl] + 160) / 400, 0, 1)
    rgb = np.stack([win] * 3, -1)
    mm = m[sl] if m.shape == arr.shape else np.zeros_like(win, bool)
    edge = mm & ~np.pad(mm, 1)[2:, 1:-1] | mm & ~np.pad(mm, 1)[:-2, 1:-1] | mm & ~np.pad(mm, 1)[1:-1, 2:] | mm & ~np.pad(mm, 1)[1:-1, :-2]
    rgb[edge] = [0.23, 0.36, 0.65]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((px, px)).save(out_png)


def export_patient(args) -> tuple[dict, dict] | None:
    pair, image_root, out_dir, make_crops = args
    import SimpleITK as sitk
    C = _EXP22
    R = sys.modules["exp22_run"]
    pid = pair["patient_id"]
    st = json.loads((C.OUT / f"state_A/{pid}.json").read_text())
    if st.get("status") != "completed":
        return None
    tx = sitk.ReadTransform(str(C.OUT / f"reg/{pid}.tfm"))
    bl = {int(k): v for k, v in st["baseline_lesions"].items()}
    fu = {int(k): v for k, v in st["followup_lesions"].items()}
    mask_b, mask_f = R.PATHS["A"](pid, "baseline"), R.PATHS["A"](pid, "followup")
    case = {
        "patient_id": pid, "cohort": pair["cohort"], "data_mode": "real", "source": "exp22_state_A",
        "baseline_exam": {"exam_id": f"{pid}_BL", "image_ref": _rel(pair["baseline_ct"], image_root)},
        "followup_exam": {"exam_id": f"{pid}_FU", "image_ref": _rel(pair["followup_ct"], image_root)},
        "baseline_lesions": [{"lesion_id": f"B{i}", "organ": None, "centroid_mm": v["centroid_mm"],
                              "centroid_registered_mm": list(tx.TransformPoint(tuple(v["centroid_mm"]))),
                              "volume_mm3": v["volume_mm3"], "diameter_mm": v["diameter_mm"],
                              "mask_ref": f"exp22:cut_A/{pid}_baseline.nii.gz#label={i}", "crop_ref": None} for i, v in sorted(bl.items())],
        "followup_lesions": [{"lesion_id": f"F{j}", "organ": None, "centroid_mm": v["centroid_mm"], "centroid_registered_mm": None,
                              "volume_mm3": v["volume_mm3"], "diameter_mm": v["diameter_mm"],
                              "mask_ref": f"exp22:cut_A/{pid}_followup.nii.gz#label={j}", "crop_ref": None} for j, v in sorted(fu.items())],
        "registration": {"transform_ref": f"exp22:reg/{pid}.tfm", "method": "exp22 rigid Euler3D, Mattes MI (frozen)", "quality": {}},
        "qc_flags": [], "tool_versions": {"segmentation": "exp22 setting A (cut_A)", "registration": "exp22 frozen rigid"},
    }
    from repair.matching_features import registration_residual
    case["registration"]["quality"] = {"residual_mm": registration_residual(case), "definition": "median NN distance of registered BL centroids to FU centroids (GT-free)"}
    if make_crops:
        for l, (mask, img, tp) in [(l, (mask_b, pair["baseline_ct"], "baseline")) for l in case["baseline_lesions"]] + \
                                   [(l, (mask_f, pair["followup_ct"], "followup")) for l in case["followup_lesions"]]:
            rel = f"crops/{pid}/{tp}_{l['lesion_id']}.png"
            _crop(img, str(mask), int(l["lesion_id"][1:]), l["centroid_mm"], Path(out_dir) / rel)
            l["crop_ref"] = rel
    ref = C.reference(pid, "A")
    mb, mf, gclass = ref["mb"], ref["mf"], ref["gclass"]
    pairs = [(f"B{a}", f"F{b}") for a in bl for b in fu if a in mb and b in mf and mb[a] == mf[b]]
    pb, pf = {a for a, _ in pairs}, {b for _, b in pairs}

    def cls_b(a):
        if f"B{a}" in pb:
            return "persistent"
        g = mb.get(a)
        return "spurious" if g is None else "disappearing" if gclass.get(g) == "disappearing" else "partner_undetected"

    def cls_f(b):
        if f"F{b}" in pf:
            return "persistent"
        g = mf.get(b)
        return "spurious" if g is None else "newly_appearing" if gclass.get(g) == "newly_appearing" else "partner_undetected"

    gt = {"patient_id": pid, "pairs": [list(p) for p in pairs], "baseline_class": {f"B{a}": cls_b(a) for a in bl},
          "followup_class": {f"F{b}": cls_f(b) for b in fu},
          "change": {f"{a}|{b}": ref["gchg"].get(mb[int(a[1:])]) for a, b in pairs if ref["gchg"].get(mb[int(a[1:])])},
          "burden": ref["gburden"], "source": "exp22b_common.reference(pid, 'A') — frozen Experiment 2.2 definitions"}
    return case, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp22-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--crops-for", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    _init(a.exp22_dir)
    pairs = _EXP22.pairs_all()[: a.limit]
    jobs = [(p, a.image_root, a.out, p["patient_id"] in set(a.crops_for)) for p in pairs]
    with Pool(a.workers, initializer=_init, initargs=(a.exp22_dir,)) as pool:
        res = [r for r in pool.map(export_patient, jobs, chunksize=2) if r]
    cases, gts = [r[0] for r in res], [r[1] for r in res]
    write_bundle(a.out, "exp22_longitudinal_ct", "real", cases, gts,
                 extra={"source": "Experiment 2.2 frozen outputs (state_A, reg, cut_A) + exp22b_common.reference",
                        "n_skipped": len(pairs) - len(res)})
    print(f"bundle written: {a.out} ({len(cases)} patients, {len(pairs) - len(res)} skipped)")


if __name__ == "__main__":
    main()
