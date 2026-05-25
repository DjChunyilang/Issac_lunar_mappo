# First-Stage Proxy Rover Asset

The first-stage implementation intentionally uses a proxy rover state model instead of a real
USD/URDF articulation. This keeps the planning, observation, reward, termination, and MAPPO
training loop testable while the real lunar rover asset and low-level control interface remain
undefined in the design documents.

Future replacement points:

- put USD assets under `usd/`
- put URDF assets under `urdf/`
- put articulation/control metadata under `cfg/`

