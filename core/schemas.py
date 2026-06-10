from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MemoryEntry:
    memory_id: str
    repo: str
    organization: str
    scope_field: str
    timestamp: float
    sensitivity: str
    license_field: str
    predicate: str
    evidence: str
    content_hash: str
    text: str


@dataclass
class ProbeContext:
    repo: str
    organization: str
    timestamp: float
    task_type: str
    language: str
    policy: str
    capability: str


@dataclass
class SequenceCard:
    sequence_id: str
    # Repository fields
    repo_url: str
    repo_commit: str
    repo_license: str
    # Task fields
    task_type: str
    prompt_hash: str
    files: List[str]
    # Memory fields
    memory_type: str
    channel: str
    evidence: str
    # Oracle fields
    oracle_type: str
    tests: List[str]
    rules: str
    policy: str
    # Intervention fields
    conditions: List[str]
    placebo_match: str
    # Annotation fields
    scope_label: str
    staleness_label: str
    bad_label: str
    security_label: str
    # Reproducibility fields
    docker_image: str
    hashes: Dict[str, str]
    seeds: List[int]


@dataclass
class RunManifest:
    sequence_id: str
    condition: str
    agent: str
    model: str
    seed: int
    temperature: float
    top_k: int
    prompt_tokens: int
    tool_calls: int
    test_calls: int
    timeout: int
    docker_digest: str
    repo_commit: str
    ledger_hash: str
    prompt_hash: str
    certificate_hash: str
    patch_hash: str
    test_log_hash: str
    pass_label: bool
    bad_label: bool
    latency: float
    cost: float
