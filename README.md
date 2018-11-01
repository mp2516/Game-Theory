# Game Theory

## Summary

This game-theory simulator can play a number of different games:
1) Rock-Paper-Scissors
2) Prisoner's Dilemma
3) Hawk-Dove

Rock-Paper-Scissors is a cyclical game which results in oscillitory strategies.

The Prisoner's Dilemma demonstrates how simple rules can lead to the emergence of widespread cooperation, despite the Defection strategy dominating each individual interaction game. However, it is also interesting for another reason: it is known to be sensitive to the activation regime employed in it.

## How to Run

##### Web based model simulation

To run the model interactively, run ``mesa runserver`` in this directory.


## Files

* ``run.py`` is the entry point for the front-end simulations.
* ``Game-Theory-Sim``: contains the model and agent classes as well as the tools needed for visualisation; the model takes a ``schedule_type`` string as an argument, which determines what schedule type the model uses: Sequential, Random or Simultaneous.

## Further Reading

This model is adapted from:

Wilensky, U. (2002). NetLogo PD Basic Evolutionary model. http://ccl.northwestern.edu/netlogo/models/PDBasicEvolutionary. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

The Demographic Prisoner's Dilemma originates from:

[Epstein, J. Zones of Cooperation in Demographic Prisoner's Dilemma. 1998.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.8629&rep=rep1&type=pdf)