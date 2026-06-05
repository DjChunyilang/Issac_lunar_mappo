# Roadmap

## Immediate

1. Keep exp008 as the current validated 3-seed terrain-aware proxy result.
2. Treat exp009 as a strong-terrain diagnostic, not a strict success.
3. Diagnose seed31 failure episodes from exp009.
4. Prototype a success-region control/reward change before further long training.

## Near Term

- Add explicit metrics for why success hold fails: `dmax_ok`, `dispersion_ok`, `speed_ok`, and hold count distribution.
- Add focused rollout debug plots for failed episodes.
- Compare reward/control variants with short seed31 runs before launching 10M+ budgets.
- Keep PhysX as validation/showcase, not the main training loop.

## Longer Term

- Revisit the action representation if success-region stability remains brittle.
- Add a stricter curriculum for terrain strength if direct strong-terrain training remains unstable.
- Build a repeatable report generator that reads `_suite/metrics/*.json` and updates experiment docs.

