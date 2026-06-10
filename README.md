# MemTrace-Bench: Code Artifact

This repository contains the code artifact for the paper **"Memory Is a Hidden Dependency: A Benchmark for Replay-Defined Harm in Stateful Coding Agents"** (ICSE 2027 submission).

## Overview

MemTrace-Bench v5 is a benchmark for evaluating persistent-memory dependencies through auditable replay. It contains 4,200 prelude-probe sequences from 1,260 repositories; 90.5% are locally runnable through public real, sanitized, or synthetic-twin releases.

### Key Results (from Paper)

| Condition | Pass Rate | Bad Rate |
|-----------|-----------|----------|
| Clean (no memory) | 61.9% ± 1.3 | 4.7% ± 0.5 |
| Warm in-scope | 75.5% ± 1.1 | 5.9% ± 0.6 |
| Warm cross-repo | 62.8% ± 1.5 | 22.6% ± 1.2 |
| Warm stale API | 64.0% ± 1.4 | 18.9% ± 1.1 |
| Warm stale security | 61.7% ± 1.6 | 28.4% ± 2.0 |
| Reference mediator | 73.1% ± 1.2 | 6.5% ± 0.7 |

## Repository Structure

```
code/
├── core/               # Core framework (estimands, schemas, conditions, predicates)
├── tables/             # Table generation scripts (Tables 1-5 from paper)
├── figures/            # Figure generation scripts (Figures 3-6 from paper)
├── baselines/          # Baseline memory system implementations
├── mediator/           # Reference mediator (lattice, compiler, certificates)
├── agents/             # Agent implementations (GitHub, SWE-bench, ReAct)
├── experiments/        # Experiment runners (replay, real-world)
├── scripts/            # Utility scripts (preprocessing, validation, generation)
├── stats/              # Statistical analysis (mixed effects)
├── lean/               # Lean 4 formal verification artifact
├── config/             # Configuration files
└── data/               # Sample data, sequences, and generated tables
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Generate Tables (using paper reference data)
```bash
python tables/table1.py --use-mock --output data/results/tables/table1.tex
python tables/table2.py --use-mock --output data/results/tables/table2.tex
python tables/table3.py --use-mock --output data/results/tables/table3.tex
python tables/table4.py --use-mock --output data/results/tables/table4.tex
python tables/table5.py --use-mock --output data/results/tables/table5.tex
```

### Run Experiments (requires API keys)
```bash
# Set up API keys
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1

# Run replay experiment
python experiments/run_replay.py \
  --sequences data/sample_sequences.json \
  --conditions clean,warm,delete-target \
  --agent-type github \
  --use-real True \
  --max-sequences 10
```

### Generate Mock Benchmark
```bash
python scripts/generate_mock_benchmark.py \
  --output data/processed/benchmark_v1.json \
  --n-sequences 4200
```

## Benchmark Composition (Paper Table 1)

| Dimension | Category | Sequences | % |
|-----------|----------|-----------|---|
| Release | public real | 1,760 | 41.9 |
| Release | sanitized executable | 1,180 | 28.1 |
| Release | synthetic twin | 860 | 20.5 |
| Release | remote-only | 400 | 9.5 |
| Memory | in-scope useful | 1,000 | 23.8 |
| Memory | cross-repo | 900 | 21.4 |
| Memory | stale dep./API | 820 | 19.5 |
| Memory | stale security | 540 | 12.9 |
| Memory | sensitive/license | 440 | 10.5 |
| Memory | prompt injection | 500 | 11.9 |

## Lean Verification Artifact

The `lean/` directory contains a bounded formal artifact verifying the reference mediator compiler:
- **86 theorems** covering policy semantics, scope/time checker, sensitivity/license, prompt compiler, certificate validator, obligation planner, channel mediator, and hash binding
- **6,090 Lean LOC** total

The verified statement: if a prompt is produced by the mediator and its certificate validates, then every factual memory segment satisfies the declared exposure policy under the mediated-channel assumption.

## Experimental Parameters

- **K = 10** repeated runs per sequence per condition
- **delta = 0.2** threshold for flagging
- **epsilon = 0.1** placebo-clean equivalence margin
- **10,000 bootstrap resamples** for confidence intervals
- **15 memory configurations** evaluated
- **6 agent families** tested
- **Multiple model families**: GPT-4.1, GPT-4o, Claude, Gemini Pro, DeepSeek-V3/R1, Qwen2.5-Coder-32B, Llama

## Dose-Response Results

| Invalid memories | Bad rate |
|-----------------|----------|
| 0 | 4.9% |
| 1 | 17.6% |
| 2 | 23.8% |
| 4 | 31.5% |

## Citation

If you use MemTrace-Bench in your research, please cite:
```bibtex
@inproceedings{memtrace2027,
  title={Memory Is a Hidden Dependency: A Benchmark for Replay-Defined Harm in Stateful Coding Agents},
  author={Anonymous},
  booktitle={Proceedings of ICSE},
  year={2027}
}
```

## License

This code is released under the MIT License. See LICENSE for details.
