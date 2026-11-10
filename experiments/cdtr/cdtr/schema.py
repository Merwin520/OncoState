"""Typed records of CDT-R. Raw evidence, proposed interpretations, independently verified (reference-derived) facts and
accepted state are separate types; generated text never becomes a measurement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Verification statuses (the decision-equivalence verifier never reports anything else)
PROVED_EQUIVALENT = 'PROVED_EQUIVALENT'
COUNTEREXAMPLE_FOUND = 'COUNTEREXAMPLE_FOUND'
INCONCLUSIVE_TIMEOUT = 'INCONCLUSIVE_TIMEOUT'
INCONCLUSIVE_INCOMPLETE_SET = 'INCONCLUSIVE_INCOMPLETE_SET'
INVALID_CLINICAL_INPUT = 'INVALID_CLINICAL_INPUT'

# Stopping actions (there is deliberately no action that rewrites history)
ACCEPT = 'accept_rule_derived_decision'
MORE_EVIDENCE = 'request_more_evidence'
DEFER = 'defer_to_clinician'
OUT_OF_BUDGET = 'out_of_budget'


@dataclass(frozen=True)
class LesionObservation:
    lesion_id: str
    visit: str                      # 'baseline' | 'followup'
    diameter_mm: float              # automatic-segmentation measurement (Exp 2.2 setting A), never generated text
    volume_mm3: float
    centroid_mm: tuple              # registered into follow-up space for baseline lesions
    provenance: str                 # source mask reference


@dataclass(frozen=True)
class VisitObservation:
    visit: str
    exam_id: str
    lesions: tuple                  # tuple[LesionObservation, ...]


@dataclass(frozen=True)
class PatientCase:
    patient_id: str                 # pseudonymous TCIA-derived id
    split: str
    baseline: VisitObservation
    followup: VisitObservation
    effective_time: str = 'baseline -> follow-up (two-visit CT)'
    registration_residual_mm: Optional[float] = None


@dataclass(frozen=True)
class MatchHypothesis:
    pairs: tuple                    # sorted tuple of (baseline_id, followup_id)
    energy: float                   # relative energy (0 = all lesions unmatched)
    source: str                     # 'V0' | 'milp_counterexample' | 'constrained_best' ...


@dataclass(frozen=True)
class CandidateState:
    hypothesis: MatchHypothesis
    response: str                   # rule-derived overall response (official engine)
    rule_inputs: dict               # exact inputs passed to the RECIST engine


@dataclass
class DecisionCheck:
    patient_id: str
    status: str
    reference_response: str         # y0 = response of the current best candidate
    q: float
    reachable_responses: list = field(default_factory=list)
    counterexample: Optional[CandidateState] = None
    witness_pairs_removed: list = field(default_factory=list)
    witness_pairs_added: list = field(default_factory=list)
    witness_lesions: list = field(default_factory=list)
    counterexample_gap: Optional[float] = None
    solver_seconds: float = 0.0
    solver_log: list = field(default_factory=list)


@dataclass(frozen=True)
class ToolAction:
    patient_id: str
    lesion_id: str
    tool: str                       # 'reference_identity_lookup' (SIMULATED review from reference annotation)
    cost: float
    policy: str
    step: int


@dataclass(frozen=True)
class EvidenceUpdate:
    patient_id: str
    lesion_id: str
    partner_id: Optional[str]       # reference partner of the queried lesion (None = no partner)
    source: str                     # 'reference annotation (simulated review; NOT clinician review)'
    arrival_step: int


@dataclass(frozen=True)
class StateCommit:
    patient_id: str
    version: int
    parent_hash: Optional[str]
    kind: str                       # 'hypothesis' | 'evidence' | 'decision'
    payload: dict
    hash: str
