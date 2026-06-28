# Method

LeWM-Flash is a measured optimization ladder for LeWorldModel's MPC planner on
Push-T. The final accepted branch is not one trick; it combines several
independently measured changes.

## Final Stack

```text
L1: cache current/goal embeddings
L2: remove loop-internal CEM CPU synchronization
L4: compile JEPA.get_cost_from_embeddings()
L6R: use TF32/high matmul precision for model-cost GEMMs
L7: bucketed iCEM with sample decay and compile shape prewarm
L7h: BF16 autocast only inside model-cost rollout
```

## Accepted Configuration

```text
solver=icem
samples=300
cem_steps=30
topk=30
horizon=5
receding_horizon=5
elite_retention=5
sample_decay=1.25
sample_count_bucket_size=60
compile_cost=true
compile_warmup_sample_counts=auto
matmul_precision=high
model_cost_autocast=bf16
```

BF16 is deliberately scoped only to `JEPA.get_cost_from_embeddings()`. The
returned planner cost tensor is cast back to `float32`, so iCEM ranking and
distribution updates keep the normal dtype contract.

## Measurement Scope

The headline planner number uses external CUDA-event timing around planner
calls, with warmup-marked plan rows excluded. It includes the real Python policy
path and action materialization around `policy.get_action()`, but not environment
stepping outside the policy.

The 3x100 full sweep has a systematic non-warmup plan-index-26 spike in every
seed. The headline `4.01x` planner speedup keeps those spikes because they are
present in both baseline and optimized runs. A conservative normal-planning view
that excludes plan index 26 gives `2.15x`.
