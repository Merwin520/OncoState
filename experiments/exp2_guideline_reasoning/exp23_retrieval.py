"""Experiment 2.3 guideline retrieval: one web-search retrieval per RECIST 1.1 clause (cached), plus mechanical verification.

Retrieval uses the same mechanism as the repository's WorkflowAgent._fetch_guidelines (OpenAI Responses API with the web_search tool).
A clause counts as
  citation_correct   : at least one cited source is the RECIST 1.1 guideline (authoritative domain or RECIST 1.1 title);
  citation_supported : the retrieved passage contains the required elements of that rule (thresholds / reference sum), checked by pattern.
These are mechanical checks, not expert adjudication.
"""
import json, os, re, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/exp23_final'
MODEL = os.environ.get('OPENAI_MODEL_SEARCH', 'gpt-5.2')
TRUSTED = ('dctd.cancer.gov', 'recist.eortc.org', 'eortc.org', 'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'sciencedirect.com', 'ejcancer.com', 'cancer.gov')
TITLE_RE = re.compile(r'response evaluation criteria in solid tumou?rs|recist', re.I)
SUFFIX = ' Answer in three sentences or fewer, state the exact numeric criteria, and cite the primary RECIST 1.1 source.'
CLAUSES = {
    'R.T.PD': ('RECIST 1.1 progressive disease for target lesions: percentage increase in sum of diameters, which reference sum is used, and the required absolute increase in mm.', [r'20\s*%', r'5\s*mm', r'(smallest|nadir)']),
    'R.T.PR': ('RECIST 1.1 partial response for target lesions: percentage decrease in sum of diameters and the reference sum used.', [r'30\s*%', r'baseline']),
    'R.T.CR': ('RECIST 1.1 complete response for target lesions: disappearance of all target lesions and the short-axis size rule for pathological lymph nodes.', [r'disappear', r'10\s*mm']),
    'R.T.SD': ('RECIST 1.1 stable disease for target lesions: definition.', [r'stable disease', r'(sufficient|neither|smallest)']),
    'R.O.NEW': ('RECIST 1.1 new lesions: effect on overall response when an unequivocal new lesion appears.', [r'new lesion', r'(progress|pd)']),
    'R.O.NTPD': ('RECIST 1.1 non-target lesions: what counts as progression of non-target disease and its effect on overall response.', [r'unequivocal', r'non-target']),
    'R.O.TPD': ('RECIST 1.1 overall response table: overall response when target lesions show progressive disease.', [r'(progressive disease|pd)', r'target']),
    'R.O.CR': ('RECIST 1.1 overall response table: overall response when target lesions show complete response, and non-target lesions show complete response, with no new lesions.', [r'complete response|cr', r'non-target']),
    'R.O.PR': ('RECIST 1.1 overall response table: overall response when target lesions show partial response or complete response and non-target lesions are non-CR/non-PD, with no new lesions.', [r'non-cr', r'(partial response|pr)']),
    'R.O.SD': ('RECIST 1.1 overall response table: overall response when target lesions are stable disease and non-target lesions are non-CR/non-PD, with no new lesions.', [r'(stable disease|sd)', r'non-cr']),
}


def main():
    from openai import OpenAI
    client = OpenAI(); out = {'model': MODEL, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'clauses': {}}
    for rid, (q, pats) in CLAUSES.items():
        t0 = time.time(); text, cites, err = '', [], None
        for attempt in range(3):
            try:
                r = client.responses.create(model=MODEL, tools=[{'type': 'web_search'}], input=q + SUFFIX, max_output_tokens=2000)
                text = r.output_text or ''
                for it in r.output:
                    for ct in (getattr(it, 'content', None) or []):
                        for a in (getattr(ct, 'annotations', None) or []):
                            if getattr(a, 'type', '') == 'url_citation':
                                cites.append({'title': getattr(a, 'title', ''), 'url': (getattr(a, 'url', '') or '').split('?utm_')[0]})
                break
            except Exception as e:
                err = type(e).__name__; time.sleep(3)
        low = text.lower()
        src_ok = any(any(d in c['url'] for d in TRUSTED) or TITLE_RE.search(c.get('title') or '') for c in cites)
        el_ok = all(re.search(p, low) for p in pats)
        out['clauses'][rid] = {'query': q, 'text': text, 'citations': cites, 'citation_correct': bool(src_ok), 'citation_supported': bool(el_ok and text), 'error': err, 'seconds': round(time.time() - t0, 1)}
        print(rid, 'source_ok', src_ok, 'supported', el_ok, len(cites), 'cites', flush=True)
    (OUT / 'guideline_retrieval.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
