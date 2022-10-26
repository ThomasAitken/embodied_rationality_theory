## Overview
The "Class 1" models of Embodied Rationality Theory are deterministic, in the sense that the agent knows exactly what each investment will give them (or, equivalently, models the world as if it knows this). In the V1 model, there is no distinction between short-term and long-term; in the V2 model, we introduce time-discounting as a rational response to environmental features. In the V2 model, we additionally introduce a parameter for the energetic depletion rate of the environment (think the difference between a rainforest and a desert).

Both Class 1 models are optimally solvable using dynamic programming (but not in a remotely efficient manner).

### Class 1 V1 Model
The Class 1 V1 model represents the world as a set of investments, each with three key parameters: a fixed resource_capacity (C), a fixed capacity_recovery_rate (r) and a fixed reward_period (p). Each additionally has a resources_to_reward function (f) and a resources_to_resources function (g).

At a given timestep during the simulation, the V1 agent determines, for a given investment, how much they will receive for a given level of resource injection via the helper method "compute_payout" which returns the following values:
- the max amount of the resource injection that can actually be put into the investment, given the current investment capacity
- the reward that should be discharged (= the reward that the resources are worth, given the level that previous investment has built up)
- the resources actually spent by the agent in this given action

The agent's goal, in this model, is simple: maximise their total lifetime reward via choosing an investment at each time step, given their starting level of resources. The different approaches to do this are implemented in algorithms.py.

#### Class 1 V2 Model
The Class 1 V2 Model extends the above model via introducing extra environment parameters that induce a role for time-discounting and alter the calculus of resource consumption.
