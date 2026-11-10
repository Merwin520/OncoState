"""Experiment 2.2 external comparison: standardized LONGITUDINAL VISUAL PACKETS for the direct multimodal models (and the generic agent).

Every patient gets the same, pre-frozen packet:
  * one coronal overview per study (mean projection) with the anonymous candidate ids placed at the lesion positions;
  * lesion-centred tiles: axial | coronal | sagittal, fixed physical field of view (80 mm), fixed 160 px panels, fixed CT window rule;
  * baseline candidates B1..Bn and follow-up candidates F1..Fm. Ids are random permutations (independent for the two studies), so an id
    never reveals a correspondence; no arrows, lines, statuses, sizes, answers or reference labels appear anywhere.
Lesion positions are the reference lesion locations (centroids of the reference masks) - the same 'physician-initialised positions'
used by Experiment 2.2 setting A. The private id->reference map is stored separately and is never shown to any model.
"""
import argparse, hashlib, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

import exp22b_common as C
import exp22_run as R
import exp22_eval as E

BASE = C.BASE.parent / 'exp22_external_final'
PK = BASE / 'packets'
CONFIG = dict(fov_mm=80.0, panel_px=160, spacing_mm=0.5, views=['axial', 'coronal', 'sagittal'], tiles_per_page=8, tile_cols=2,
              window_rule='mean HU in the central 7 mm cube < -300 -> lung window (L -600, W 1500); otherwise soft-tissue window (L 40, W 400)',
              lung_window=(-600.0, 1500.0), soft_window=(40.0, 400.0), overview=dict(window_level=-300.0, window_width=1400.0, width_px=384, max_height_px=512, projection='mean over the anterior-posterior axis'),
              scale_bar_mm=20, interpolation='linear', outside_value_hu=-1024, ids='B1..Bn (baseline), F1..Fm (follow-up), random permutation per study', reference_positions='centroids of reference lesion masks')
DIRS = {'axial': ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 'coronal': ((1, 0, 0), (0, 0, -1), (0, 1, 0)), 'sagittal': ((0, 1, 0), (0, 0, -1), (1, 0, 0))}
LABELS = {'axial': ('R', 'L', 'A', 'P'), 'coronal': ('R', 'L', 'S', 'I'), 'sagittal': ('A', 'P', 'S', 'I')}
try: FONT = ImageFont.load_default(size=11)
except Exception: FONT = ImageFont.load_default()


def seed(pid, tp): return int(hashlib.sha256(f'exp22x|{pid}|{tp}'.encode()).hexdigest()[:8], 16)


def plane(ct, c, view, fov, px, sp):
    (a, b, n) = DIRS[view]; a, b, n = map(np.array, (a, b, n))
    origin = np.array(c, float) - a * fov / 2 - b * fov / 2
    M = np.stack([a, b, n], 1); ref = sitk.Image(px, px, 1, sitk.sitkFloat32)
    ref.SetSpacing((float(sp), float(sp), float(sp))); ref.SetOrigin(tuple(float(x) for x in origin)); ref.SetDirection(tuple(float(x) for x in M.ravel()))
    out = sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, CONFIG['outside_value_hu'])
    return sitk.GetArrayFromImage(out)[0]


def window(img, L, W): return np.clip((img - (L - W / 2)) / W, 0, 1)


def central_mean(ct, c):
    ref = sitk.Image(14, 14, 14, sitk.sitkFloat32); ref.SetSpacing((0.5, 0.5, 0.5)); ref.SetOrigin(tuple(float(x) for x in (np.array(c, float) - 3.5)))
    return float(sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, CONFIG['outside_value_hu'])).mean())


def tile(ct, c, cid):
    px, fov, sp = CONFIG['panel_px'], CONFIG['fov_mm'], CONFIG['spacing_mm']
    hu = central_mean(ct, c); lung = hu < -300; L, W = CONFIG['lung_window'] if lung else CONFIG['soft_window']
    img = Image.new('L', (3 * px, px + 16), 0); d = ImageDraw.Draw(img)
    d.text((3, 2), f'{cid}  ({"lung" if lung else "soft-tissue"} window, panels 80 mm)', fill=255, font=FONT)
    for k, v in enumerate(CONFIG['views']):
        p = (window(plane(ct, c, v, fov, px, sp), L, W) * 255).astype(np.uint8); pi = Image.fromarray(p); img.paste(pi, (k * px, 16)); dd = ImageDraw.Draw(img)
        l, r, t, b = LABELS[v]; x0 = k * px
        dd.text((x0 + 2, 16 + px // 2 - 5), l, fill=255, font=FONT); dd.text((x0 + px - 9, 16 + px // 2 - 5), r, fill=255, font=FONT)
        dd.text((x0 + px // 2 - 3, 17), t, fill=255, font=FONT); dd.text((x0 + px // 2 - 3, 16 + px - 12), b, fill=255, font=FONT)
        if v == 'axial': dd.line((x0 + 8, 16 + px - 8, x0 + 8 + int(CONFIG['scale_bar_mm'] / sp), 16 + px - 8), fill=255, width=2)
        dd.rectangle((x0, 16, x0 + px - 1, 16 + px - 1), outline=90)
    return img, ('lung' if lung else 'soft')


def pages(tiles):
    n, cols = CONFIG['tiles_per_page'], CONFIG['tile_cols']; out = []
    for s in range(0, len(tiles), n):
        chunk = tiles[s:s + n]; w, h = chunk[0].size; rows = (len(chunk) + cols - 1) // cols
        page = Image.new('L', (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), 40)
        for i, t in enumerate(chunk): page.paste(t, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
        out.append(page)
    return out


def overview(ct, pts, ids):
    arr = sitk.GetArrayFromImage(ct); sp = ct.GetSpacing(); o = CONFIG['overview']
    proj = arr.mean(axis=1)                                                   # (z, x) mean over the anterior-posterior axis
    img = window(proj, o['window_level'], o['window_width'])[::-1]            # superior at top
    zmm, xmm = img.shape[0] * sp[2], img.shape[1] * sp[0]; wpx = o['width_px']; hpx = int(min(o['max_height_px'], wpx * zmm / xmm)); scale_x = wpx / img.shape[1]; scale_y = hpx / img.shape[0]
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((wpx, hpx), Image.BILINEAR).convert('RGB'); d = ImageDraw.Draw(im)
    for (x, y, z), cid in zip(pts, ids):
        i = ct.TransformPhysicalPointToContinuousIndex((x, y, z)); px_, py_ = i[0] * scale_x, (img.shape[0] - 1 - i[2]) * scale_y
        d.ellipse((px_ - 3, py_ - 3, px_ + 3, py_ + 3), outline=(255, 60, 60)); d.text((px_ + 4, py_ - 6), cid, fill=(255, 220, 60), font=FONT)
    d.text((3, 2), 'R', fill=(255, 255, 255), font=FONT); d.text((wpx - 10, 2), 'L', fill=(255, 255, 255), font=FONT)
    return im


def build(args):
    pid, p = args
    try:
        gb, gf = E.gt_img(pid, 'baseline'), E.gt_img(pid, 'followup'); recs = {}
        for tp, gimg, ctp in (('baseline', gb, p['baseline_ct']), ('followup', gf, p['followup_ct'])):
            d = R.describe(gimg)[1]; gids = sorted(d); rng = np.random.default_rng(seed(pid, tp)); order = rng.permutation(len(gids))
            prefix = 'B' if tp == 'baseline' else 'F'; cid = {gids[j]: f'{prefix}{k + 1}' for k, j in enumerate(order)}      # random neutral ids
            ct = sitk.ReadImage(ctp, sitk.sitkFloat32); listing = sorted(gids, key=lambda g: int(cid[g][1:])); tiles, wins = [], {}
            for g in listing:
                t, w = tile(ct, d[g]['centroid_mm'], cid[g]); tiles.append(t); wins[g] = w
            paths = []
            for k, pg in enumerate(pages(tiles)):
                f = PK / pid / f'{tp}_lesions_page{k + 1}.png'; f.parent.mkdir(parents=True, exist_ok=True); pg.save(f); paths.append(str(f.relative_to(BASE)))
            f = PK / pid / f'{tp}_overview.png'; overview(ct, [d[g]['centroid_mm'] for g in listing], [cid[g] for g in listing]).save(f)
            recs[tp] = dict(ids={cid[g]: int(g) for g in listing}, pages=paths, overview=str(f.relative_to(BASE)), windows={cid[g]: wins[g] for g in listing}, centroids={cid[g]: d[g]['centroid_mm'] for g in listing})
        (PK / pid / 'private_key.json').write_text(json.dumps(dict(patient_id=pid, baseline=recs['baseline']['ids'], followup=recs['followup']['ids'])))
        (PK / pid / 'public_packet.json').write_text(json.dumps(dict(patient_id=pid, study_pair='baseline->followup', **{tp: dict(candidate_ids=sorted(recs[tp]['ids'], key=lambda s: int(s[1:])), pages=recs[tp]['pages'], overview=recs[tp]['overview']) for tp in recs})))
        rows = [dict(patient_id=pid, study_pair='baseline->followup', timepoint=tp, baseline_lesion_id=cid_ if tp == 'baseline' else '', candidate_id=cid_, views='axial;coronal;sagittal', crop_size_mm=CONFIG['fov_mm'],
                     window=recs[tp]['windows'][cid_], image_paths=';'.join(recs[tp]['pages'] + [recs[tp]['overview']]), generation_status='ok') for tp in recs for cid_, g in recs[tp]['ids'].items()]
        return rows
    except Exception as e:
        return [dict(patient_id=pid, study_pair='baseline->followup', generation_status=f'failed: {type(e).__name__}: {e}')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', default='locked'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    pairs = [p for p in C.pairs_all() if p.get('cohort') == a.cohort]
    if a.limit: pairs = pairs[:a.limit]
    BASE.mkdir(parents=True, exist_ok=True)
    with Pool(12) as pool: res = pool.map(build, [(p['patient_id'], p) for p in pairs], chunksize=1)
    rows = [r for x in res for r in x]; df = pd.DataFrame(rows)
    name = 'visual_packet_manifest.csv' if a.cohort == 'locked' else f'visual_packet_manifest_{a.cohort}.csv'
    df.to_csv(BASE / name, index=False); (BASE / 'visual_packet_config.json').write_text(json.dumps(CONFIG, indent=1))
    print(len(pairs), 'patients;', int((df.generation_status == 'ok').sum()), 'candidates ok;', int((df.generation_status != 'ok').sum()), 'failures')


if __name__ == '__main__':
    main()
