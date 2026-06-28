# Method

LeWM-Flash is a measured optimization ladder for LeWorldModel's MPC planner on
Push-T. The final accepted branch is not one trick; it combines several
independently measured changes.

## Final Configuration

LeWM-Flash combines six scoped changes:

1. Reuse the current-observation and goal embeddings across repeated
   model-cost calls inside one planning query.
2. Keep CEM iteration bookkeeping on the GPU instead of synchronizing candidate
   costs back to the CPU on every optimizer step.
3. Route repeated model-cost evaluation through a tensor-only helper,
   `JEPA.get_cost_from_embeddings()`, and compile that helper.
4. Request PyTorch's `high` float32 matmul precision so the model-cost GEMMs can
   use TF32 tensor cores on NVIDIA hardware that supports them.
5. Replace exact CEM with bucketed iCEM: candidate counts decay as the sampling
   distribution narrows, but they are rounded into stable buckets for compile
   reuse.
6. Run only the learned model-cost rollout under CUDA BF16 autocast, then return
   costs to `float32` before iCEM ranking and distribution updates.

The historical experiment ledger maps these to `L1`, `L2`, `L4`, `L6R`, `L7`,
and `L7h`; those labels are retained only so the public artifact can be traced
back to the original measurements.

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
