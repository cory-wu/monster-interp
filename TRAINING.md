# Monster Gridworld training

The training stack is intentionally split by responsibility:

- `monster_agent.py`: convolutional recurrent actor-critic and PopArt value head.
- `vmpo.py`: discrete V-MPO objective and discounted returns.
- `rollout_collector.py`: synchronous batched environment interaction.
- `train_vmpo.py`: configuration, optimization, metrics, and checkpoints.
- `evaluate_agent.py`: reward curves and behavioral evaluation reports.

## One-million-step pilot

```bash
python train_vmpo.py \
  --steps 1000000 \
  --seed 42 \
  --n-envs 32 \
  --run-dir runs/vmpo_seed42_1m
```

Each run directory contains CSV metrics, checkpoints, a reward curve, and behavior
summaries for the 25-step training environment, a 200-step environment, and a
200-step no-monsters counterfactual. Both sampled and argmax policies are evaluated.

## Continue the pilot to ten million steps

`--steps` is the cumulative target, so this extends the existing run rather than
adding ten million more steps:

```bash
python train_vmpo.py \
  --steps 10000000 \
  --seed 42 \
  --n-envs 32 \
  --run-dir runs/vmpo_seed42_1m \
  --resume-from runs/vmpo_seed42_1m/checkpoint_final.pt
```

Use a new run directory for a different seed or episode length. Training refuses
to overwrite existing metrics unless `--resume-from` is supplied.
