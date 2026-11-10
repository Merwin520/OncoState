"""Bind the UNCHANGED Exp 1 agents (exp22b_agent = stateless harness, stateful_baseline/generic_stateful_agent = generic stateful agent)
to MU-Glioma-Post visit evidence.

Only the dataset-specific attributes of the exp22b_agent module are replaced: SYSTEM prompt, TOOLS, QUERIES, validate, and the two
Toolbox methods (__init__, run). Both agents look these up at call time, so:
  * exp22b_agent.episode (stateless loop) and generic_stateful_agent.episode / update_memory (memory loop) run verbatim;
  * generic_stateful_agent.RecordingToolbox (a subclass of exp22b_agent.Toolbox) inherits the MRI tools without modification;
  * the model, MAX_TURNS, Responses-API loop, parser patch (local_model_patch) and memory semantics are exactly those of Exp 1.
The source files of both agents are hash-checked against the Exp 1 versions at install time.

Query ids passed to the harness are '<visit>|<Q>' or '<visit>|<Q>|late' so that the memory key logic of update_memory is untouched.
"""
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP1 = HERE.parent / 'exp1_agentic_longitudinal'
sys.path.insert(0, str(HERE))
import mu_glioma_adapter as A  # noqa: E402
import queries as Q  # noqa: E402

EXP1_SHA256 = {   # files exactly as used for the Exp 1 controlled-memory run (mtimes predate that run's raw logs)
    'stateful_baseline/generic_stateful_agent.py': 'de1c203b23e5efb9ff40ec316761b8ac8341c2ac246c2a97fc16b29214e88369',
    'exp22b_agent.py': '7094b0faac5b7ef9ff400d1b603d5d8ad35d5c7619f9f4cc77f8ad12bee8b315',
    'stateful_baseline/local_model_patch.py': '5025a3fd16664a8e22a27be10778777c818d065e941a224b342729afcc5e2cee',
}

SYSTEM = """You are a tool-using analysis agent for serial post-treatment brain MRI visits of a glioma patient.
Each visit has a released tumour segmentation that has already been measured; you work through the tools listed below. Visit ids are
strings such as "Timepoint_3"; visits are ordered chronologically as returned by list_visits. Only visits that have already arrived are
available; never assume anything about visits that are not available.

Definitions you must use:
- Disease burden (burden_ml) of a visit = the burden_ml value of that visit's record (whole tumour excluding the resection cavity, mL).
- Previous observed visit = the latest available visit before the current visit. First observed visit = the earliest available visit.
- Change = current burden - reference burden (mL); pct_change = 100 x change / reference burden (null if the reference burden is 0).
- Nadir = the minimum burden over all available visits up to and including the current visit (ties: the earliest such visit);
  current_minus_nadir_ml = current burden - nadir burden.
- Step direction at a visit (from the previous observed visit): diff = burden - previous burden, thr = max(0.10 x previous burden, 1.0 mL);
  increase if diff > thr, decrease if diff < -thr, otherwise stable.
- Reversal at the current visit: true if the current step is not stable AND the most recent earlier non-stable step exists and has the
  opposite direction; otherwise false.
- Treatment categories: surgery, radiation, initial_chemotherapy, additional_therapy, second_additional_therapy, immunotherapy,
  brachytherapy, other_therapy. Each visit record lists the treatment events reported since the previous visit; the treatments started by
  the current visit are all events reported in the records of the available visits up to and including the current visit.
- Numbers: use mL values as given (2 decimals); report computed numbers to 2 decimals.

Work only from tool outputs. When you are done, reply with ONE JSON object in the exact format requested and nothing else."""

TOOLS = [
    {'type': 'function', 'name': 'list_visits', 'description': 'The visits of a patient that are available now, in chronological order, with days from diagnosis, and the current visit.',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}}, 'required': ['patient_id']}},
    {'type': 'function', 'name': 'get_visit_record', 'description': 'The record of one available visit: burden_ml and compartment volumes (mL) from the released segmentation, days from diagnosis, and treatment events reported since the previous visit.',
     'parameters': {'type': 'object', 'properties': {'patient_id': {'type': 'string'}, 'timepoint': {'type': 'string', 'description': 'visit id, e.g. Timepoint_2'}}, 'required': ['patient_id', 'timepoint']}},
    {'type': 'function', 'name': 'calculate', 'description': 'Evaluate an arithmetic expression (numbers, + - * / **, sum, len, min, max, abs, round, list comprehensions over literal lists).',
     'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}},
]

LATE_NOTE = '{late} is a delayed study that was acquired before the current visit; it has just arrived and is now available.'


class QueryMap(dict):
    """QUERIES lookup for qid '<visit>|<Q>[|late:<visit>]' -> (query text naming the current visit, answer format)."""
    def __getitem__(self, qid):
        parts = qid.split('|')
        visit, q = parts[0], parts[1]
        text = f'Current visit: {visit}. ' + Q.TEXT[q][0]
        if len(parts) > 2 and parts[2].startswith('late:'):
            text = LATE_NOTE.format(late=parts[2][5:]) + ' ' + text
        return text, Q.TEXT[q][1]

    def __contains__(self, qid):
        return True


def _toolbox_init(self):
    self.calls = []; self.registered = set(); self.seen_ids = {'baseline': set(), 'followup': set()}; self.visits_read = []


def _toolbox_run(self, name, args):
    """Same contract as exp22b_agent.Toolbox.run (JSON string {'ok', 'result'|'error'}, every call logged with compute time)."""
    G = sys.modules['exp22b_agent']
    t0 = time.time(); compute = 0.0; read = None
    try:
        pid = str(args.get('patient_id', ''))
        if name == 'list_visits':
            ctx = A.VisitClock.get()
            vis = A.VisitClock.available_visits(pid)
            out = {'patient_id': pid, 'current_visit': ctx['current'], 'visits': [{'visit': v, 'days_from_diagnosis': A.mri_day(pid, v)} for v in vis]}
        elif name == 'get_visit_record':
            tp = str(args.get('timepoint', ''))
            ctx = A.VisitClock.check(pid, tp)                       # raises FutureAccess for visits that have not arrived
            out, compute = A.visit_record(pid, tp, ctx['visits'])
            read = tp; self.visits_read.append(tp)
        elif name == 'calculate':
            expr = str(args.get('expression', ''))
            if len(expr) > 20000 or any(not isinstance(n, G.ALLOWED_NODES) for n in ast.walk(ast.parse(expr, mode='eval'))):
                raise ValueError('expression not allowed')
            out = eval(compile(ast.parse(expr, mode='eval'), '<calc>', 'eval'), {'__builtins__': {}}, dict(G.SAFE))
        else:
            raise ValueError('unknown tool')
        res = {'ok': True, 'result': out}
    except Exception as e:
        res = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    self.calls.append(dict(name=name, args=args, seconds=round(time.time() - t0, 3), compute_seconds_if_uncached=round(compute, 3), ok=res['ok'],
                           visit_read=read, future_blocked=(not res['ok']) and 'FutureAccess' in res.get('error', '')))
    return json.dumps(res)


def file_sha(rel):
    return hashlib.sha256((EXP1 / rel).read_bytes()).hexdigest()


def install(measure_cache):
    """Import the Exp 1 agents (hash-verified), bind MU-Glioma evidence, return (exp22b_agent, generic_stateful_agent)."""
    for rel, h in EXP1_SHA256.items():
        if file_sha(rel) != h:
            raise RuntimeError(f'Exp 1 file changed: {rel}')
    os.environ.setdefault('EXP22B_MODEL', 'qwen3.5-9b')
    sys.path.insert(0, str(EXP1)); sys.path.insert(0, str(EXP1 / 'stateful_baseline'))
    import generic_stateful_agent as GSA  # noqa: E402  (imports exp22b_agent and applies local_model_patch, exactly as in Exp 1)
    G = sys.modules['exp22b_agent']
    assert GSA.G is G and issubclass(GSA.RecordingToolbox, G.Toolbox)
    G.SYSTEM, G.TOOLS, G.QUERIES = SYSTEM, TOOLS, QueryMap()
    G.validate = lambda qid, j: Q.validate(qid.split('|')[1], j)
    G.Toolbox.__init__, G.Toolbox.run = _toolbox_init, _toolbox_run
    G.PROMPT_TEXT = SYSTEM + '\n\n' + '\n'.join(f'[{k}] {q}\nFormat: {f}' for k, (q, f) in Q.TEXT.items())
    G.PROMPT_HASH = hashlib.sha256(G.PROMPT_TEXT.encode()).hexdigest()
    A.load_measure_cache(measure_cache)
    return G, GSA
