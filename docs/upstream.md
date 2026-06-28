# Upstream Provenance

LeWM-Flash is an optimization and benchmark artifact built on upstream LeWM and
stable-worldmodel code.

## LeWorldModel

- Paper: https://arxiv.org/abs/2603.19312
- Source: https://github.com/lucas-maes/le-wm
- Imported commit in this artifact lineage:
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- License: MIT, retained in `lewm/LICENSE`

Adapted files include the JEPA model implementation, model modules, eval
configs, and local solver integration points. The main performance additions are
the embedding/model-cost runtime hooks and optimized planner solvers.

## stable-worldmodel

- Source: https://github.com/galilai-group/stable-worldmodel
- Role: environment management, policy/evaluation APIs, and upstream planning
  conventions used by LeWM.
LeWM-Flash does not vendor the full stable-worldmodel source. It depends on
`stable-worldmodel[train,env]` in the Modal benchmark runtime and records the
package version in benchmark manifests.

## Planning Literature

- iCEM: https://arxiv.org/abs/2008.06389
- CEM-GD context: https://arxiv.org/abs/2112.07746

The accepted LeWM-Flash branch uses bucketed iCEM. The CEM-GD/iCEM-GD path was
implemented and measured, but not promoted.
