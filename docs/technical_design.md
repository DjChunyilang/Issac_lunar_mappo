# Technical Design Mirror

The implementation follows `isaac_sim_skrl_mappo_multi_rover_tech_doc_v2_0.md` for the first-stage
scope:

- 4 homogeneous rover agents
- decentralized actor observation
- centralized critic state
- `[rho, beta]` local subgoal action
- deterministic line trajectory generator
- simplified velocity tracking
- geometric gathering reward and oracle distance-progress reward

