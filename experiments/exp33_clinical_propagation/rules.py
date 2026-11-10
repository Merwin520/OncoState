"""Exp 3.3 shared rules: the downstream assessment (identical to Exp 3.0 `assess`, plus lesion statuses) and correction statements.
Pure Python, importable from every environment."""

RULE_TEXT = """Downstream assessment rules (apply exactly):
- Lesion status: a baseline lesion with a follow-up partner is "persistent", without one "disappeared"; a follow-up lesion with a baseline
  partner is "persistent", without one "new". Matching is one-to-one.
- Target lesions: the (at most) 5 baseline lesions with the largest diameter among those with diameter >= 10 mm.
- target_sum_baseline_mm = sum of the baseline diameters of the target lesions.
- target_sum_followup_mm = sum, over the target lesions, of the diameter of the matched follow-up partner (0 if a target lesion has no partner).
- new_lesion_ge_10mm = true if any follow-up lesion WITHOUT a baseline partner has diameter >= 10 mm.
- response: if there are no target lesions: "PD" if new_lesion_ge_10mm else "NE".
  Otherwise "PD" if new_lesion_ge_10mm or (followup sum - baseline sum >= 0.2 x baseline sum and >= 5 mm);
  else "CR" if the follow-up sum is 0; else "PR" if follow-up sum <= 0.7 x baseline sum; else "SD"."""


def assess(case, pairs):
    bl = {l['lesion_id']: l['diameter_mm'] for l in case['baseline_lesions']}
    fu = {l['lesion_id']: l['diameter_mm'] for l in case['followup_lesions']}
    pairs = [tuple(x) for x in pairs]
    partner = {a: b for a, b in pairs}
    targets = [a for a, d in sorted(bl.items(), key=lambda kv: -kv[1]) if d >= 10][:5]
    s0 = sum(bl[a] for a in targets)
    s1 = sum(fu[partner[a]] if partner.get(a) in fu else 0. for a in targets)
    matched_fu = {b for _, b in pairs}
    new = any(d >= 10 for b, d in fu.items() if b not in matched_fu)
    if not targets:
        resp = 'PD' if new else 'NE'
    elif new or (s1 - s0 >= .2 * s0 and s1 - s0 >= 5):
        resp = 'PD'
    elif s1 == 0:
        resp = 'CR'
    elif s1 <= .7 * s0:
        resp = 'PR'
    else:
        resp = 'SD'
    status = {a: ('persistent' if a in partner else 'disappeared') for a in bl}
    status.update({b: ('persistent' if b in matched_fu else 'new') for b in fu})
    return dict(response=resp, new_lesion=bool(new), sum_bl=round(s0, 1), sum_fu=round(s1, 1), status=status, pairs=sorted(pairs))


def correction_statements(v0, corrected):
    """The verified clinician correction as plain statements: only what differs between V0 and the corrected correspondences."""
    a, b = {x for x, _ in v0}, None
    p0, p1 = dict(v0), dict(corrected)
    f0, f1 = {f: x for x, f in v0}, {f: x for x, f in corrected}
    out = []
    for bl in sorted(set(p0) | set(p1)):
        if p0.get(bl) != p1.get(bl):
            out.append(f'baseline {bl} and follow-up {p1[bl]} are the same lesion (matched pair)' if p1.get(bl)
                       else f'baseline {bl} has no follow-up counterpart (disappeared)')
    for fu in sorted(set(f0) | set(f1)):
        if fu in f0 and fu not in f1:
            out.append(f'follow-up {fu} has no baseline counterpart (new lesion)')
    return out
