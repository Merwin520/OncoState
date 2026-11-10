"""Official TCIA clinical / measurement spreadsheets. Files are downloaded once, hash-recorded, and read as-is."""
import hashlib
import json
import urllib.request
import warnings

import pandas as pd

from paths import config, source_dir

warnings.filterwarnings('ignore', message='Unknown extension')


def fetch(name):
    f = source_dir() / name
    if not f.exists():
        urllib.request.urlretrieve(config('ispy2')['source_base_url'] + name, f)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    reg = source_dir() / 'SOURCES.json'
    d = json.loads(reg.read_text()) if reg.exists() else {}
    d[name] = sha
    reg.write_text(json.dumps(d, indent=1))
    return f, sha


def ispy2_clinical():
    f, sha = fetch(config('ispy2')['clinical_file'])
    x = pd.read_excel(f, sheet_name=None)
    assert len(x) == 1, f'unexpected sheets: {list(x)}'
    d = next(iter(x.values()))
    d['patient_id'] = d.Patient_ID.astype(str)
    return d, dict(file=f.name, sha256=sha, sheet=next(iter(x)))


def ispy2_measurements():
    """UCSF multi-feature table: tumour volume V10..V40 (visits 1-4), LD, sphericity, BPE at T0..T3. Only patients with 4 visits."""
    f, sha = fetch(config('ispy2')['measurement_file'])
    x = pd.read_excel(f, sheet_name=None)
    sheet = next(iter(x))
    d = x[sheet]
    d['patient_id'] = d['CLINICAL-TRIAL-SUBJECT-ID'].astype(str)
    return d, dict(file=f.name, sha256=sha, sheet=sheet)


def ispy1_clinical():
    f, sha = fetch(config('ispy1_optional')['clinical_file'])
    x = pd.read_excel(f, sheet_name=None)
    cl, oc = x['TCIA Patient Clinical Subset'], x['TCIA Outcomes Subset']
    d = cl.merge(oc.drop(columns=['DataExtractDt']), on='SUBJECTID', how='left')
    d['patient_id'] = d.SUBJECTID.astype(str)
    return d, dict(file=f.name, sha256=sha, sheets=list(x))
