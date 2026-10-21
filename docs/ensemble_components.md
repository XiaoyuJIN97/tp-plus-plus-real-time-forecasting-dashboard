# Ensemble component discovery

The ensemble now discovers components from config/model_registry.yml. Every
model that is enabled, supports the target and is not pipeline_generated is
included by default when its current run has 24 contiguous hourly predictions
with finite values. Set ensemble_include: false to opt out. Persistence and
experimental/hidden TabPFN explicitly opt out. The ensemble itself is excluded.
TSO remains controlled by each zone/target's include_tso selection.

Current ordinary members are Chronos2, Ridge (load) or XGBoost (renewables),
TimesFM3, T0 Gamma and Timestral: five models, plus TSO where selected. Future
registry additions join automatically once their forecasts are available. This
membership rule does not install or schedule inference for a newly added model;
its individual forecasting integration must also be in place.

The existing mean/median/constrained-learned choice per zone/target is retained.
Those choices were selected with the older component set; this change does not
claim they are still optimal for the expanded set. Learned weights are refitted
per six-hour horizon block using up to 60 earlier complete runs (minimum 21),
non-negative weights summing to one, a 70% cap and regularization 0.1. Prior
runs must finish delivery before the new run's origin. Actual publication/revision
vintages are not reconstructed; historical fits use the stored actuals.

Historical runs before a new model is available, or runs with temporarily
incomplete/non-finite model predictions, use the complete available members.
Omissions are logged. Fewer than two available components yields no ensemble.
All training rows must contain all selected members and actuals; models missing
in earlier runs are not filled with fake predictions or zero weights. Saved
offline weights are used only if they cover every selected component. Otherwise
warmup uses equal weights until the expanded model set has sufficient history.
Optimization failure also falls back to equal weights.

Each ensemble row records ensemble_components (model keys) and, for learned
ensembles, ensemble_weights (labels/weights by block). History repair checks the
component signature before retaining an existing forecast; membership changes
therefore regenerate it. Adding a later model does not invent historical outputs
for that model. The daily pipeline and observed-outcome refresh repair stale
membership and rebuild leaderboard and probabilistic artifacts afterward.

This task changes ensemble membership, not model licensing. Existing TimesFM3
and Timestral license conditions discussed in the license audit still apply.
