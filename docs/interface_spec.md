# Interface Spec

## Actor Observation

Shape: `(num_envs, 4, obs_dim)`.

The actor observation contains ego state, neighbor state sharing, flat-terrain handcrafted
features, and local aggregation features. It does not contain `p*`, oracle distances, or oracle
distance reductions.

## Critic State

Shape: `(num_envs, state_dim)`.

The critic state contains all rover true states, team geometry, terrain summary, and training-only
oracle features.

## Action

Shape: `(num_envs, 4, 2)`.

The normalized action is mapped to:

- `rho in [0, rho_max]`
- `beta in [-beta_max, beta_max]`

## First-Stage Dynamics

The current rover is a proxy unicycle state model. Replace it with a real Isaac Sim articulation
only after the rover asset and control interface are defined.

