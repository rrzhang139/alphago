# Experiment Queue

Research agent writes JSON configs here. Infra agent picks them up and runs on GPU.

## Protocol

1. Research agent creates `experiments/queue/<name>.json`
2. Research agent commits and pushes
3. Infra agent detects new file (via polling or notification)
4. Infra agent provisions GPU, runs experiment
5. Infra agent moves config to `done/<name>.json` with results appended
6. Infra agent pushes results (weights, history, plots)

## Queue File Format

```json
{
  "name": "go9_constant_lr",
  "priority": "high",
  "hypothesis": "Constant LR prevents loss U-shape",
  "game": "go9",
  "run_script": "experiments/20260313_go9_fix_c/run.py",
  "gpu": "A4000",
  "estimated_time": "2h",
  "requested_by": "research-agent",
  "requested_at": "2026-03-13T12:00:00Z"
}
```
