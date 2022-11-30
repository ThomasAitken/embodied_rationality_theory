## Overview
As the name suggets, this directory contains the deterministic models.

### Minimal
The minimal deterministic model is the simplest model of Embodied Rationality Theory, carrying with it the following assumptions:
- The agent only cares about reward and a single kind of resources.
- The investments follow the `discharge` pattern - they each have a distinct resource threshold, then discharge their own level of reward/resources. Then they recharge according to a distinct `capacity_recovery_rate`.
- The environment is stable, fixed and fully known by the agent: it consists of the same suite of investments at all time steps.
- The agent must stay above 0 resources to survive but the level of resources possessed by the agent has no effect on action efficiency.
- The agent is selfish.

### Minimal Unselfish
The minimal unselfish deterministic model extends the minimal model by modifying one thing: the assumption of selfishness. Each investment is modified to carry a reward/resource payoff value for each external agent the main agent cares about, and the main agent's level of concern for different agents is modelled by a fixed weighting. So the main agent now optimises to maximise this weighted sum over its lifetime, rather than a single sum.

Each investment in the environment now has reward and resources relative to the entity who benefits from it. We assume (in a less-than-fully-realistic fashion) that no investment benefits two agents at once.