## Overview
The "Class 1" models of Embodied Rationality Theory assume perfect knowledge on the part of the agent on what each investment will give them. In the V1 model, there is no distinction between short-term and long-term; in the V2 model, we introduce time-discounting as a rational response to environmental features. In the V2 model, we additionally introduce a parameter for the energetic depletion rate of the environment (think the difference between a rainforest and a desert).

In the V3 model, we disaggregate resources into "energetic" and "instrumental" in order to better model humans in particular. In V1 and V2, the concept of "resources" essentially just means "energetic resources", which is an adequate construct for modelling almost all of biology, only falling short in modelling human society. The concept of "instrumental" resources refers to the additional category of resources that exist in human society, those that act as credit for future energetic resources. These resources move around in a more complex fashion than energetic resources (because they are spatially and temporally decoupled from them), and hence need to be modelled differently.

All the Class 1 models are optimally solvable using Dynamic Programming, as proved in the accompanying paper. Yet any such algorithm is horrendously inefficient and exponential. (Note that even if biological intelligences could run such an algorithm, it probably wouldn't even be that useful given the non-realism of the assumption of perfect knowledge.) We compare the optimal approaches with several heuristic-based alternatives.

In all this modelling, the choice has been made to restrict ourselves to integer functions, i.e. functions Z -> Z. Not much changes from a theoretical point of view if we change these integers to floats, but it makes for a much simpler task of numerical computing if we use integers. Using floats wouldn't give us much in terms of model realism either, since the parameters involved in these functions are likely not measurable to any great precision in nature. 

### Class 1 V1 Model
The Class 1 V1 model represents the world as a set of investments, each with three key parameters: a fixed resource_capacity (C), a fixed capacity_recovery_rate (r) and a fixed reward_period (p). Each additionally has a resources_to_reward function (f) and a resources_to_resources function (g).

At a given timestep during the simulation, the V1 agent determines, for a given investment, how much they will receive for a given level of resource injection via the helper method "compute_payout" which returns the following values:
- the max amount of the resource injection that can actually be put into the investment, given the current investment capacity
- the reward that should be discharged (= the reward that the resources are worth, given the level that previous investment has built up)
- the resources actually spent by the agent in this given action

The agent's goal, in this model, is simple: maximise their total lifetime reward via choosing an investment at each time step, given their starting level of resources. The different approaches to do this are implemented in algorithms.py.

#### Class 1 V2 Model
The Class 1 V2 Model extends the above model via introducing extra environment parameters that induce a role for time-discounting and alter the calculus of resource consumption.

#### Class 1 V3 Model
The Class 1 V3 model disaggregates the resources parameter into energetic and instrumental resources. Whereas V1 and V2 are for modelling all organisms capable of planning, V3 is only intended to model humans, since the concept of instrumental resources basically maps to "artefacts that have a use or exchange value" (i.e. including money, valuable goods, tools, useful assets, growth assets). In addition to an energetic_depletion_rate, we now have an instrumental_resource_depletion_rate. The mapping of the resources_to_reward function for each investment remains in terms of total resources invested (energetic + instrumental), but each investment now has a minimal energetic burden per interaction. Whereas for V1 and V2, the resources_to_resources function is always non-negative for all investments (i.e. the worst resource outcome is losing all the resources expended), we now disaggregate this single function into one that preserves this original structure (the energetic_resources_to_energetic_resources function) and another which doesn't (the instrumental_resources_to_instrumental_resources function). The instrumental_resources_to_instrumental_resources function can be negative to model the situation where some action incurs some kind of debt in addition to the loss of the resources invested.
