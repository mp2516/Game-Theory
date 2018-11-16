# Game Theory

## Summary

This game-theory simulator can play a number of different games:

##### Rock-Paper-Scissors



##### Prisoner's Dilemma

The Prisoner's Dilemma demonstrates how simple rules can lead to the emergence of widespread cooperation, despite the Defection strategy dominating each individual interaction game. However, it is also interesting for another reason: it is known to be sensitive to the activation regime employed in it.


## Files

* ``run.py`` is the entry point for the front-end simulations.
* ``agent.py``: contains the agent class which dictates how the agents play each other. Here the strategies of the agents are determined and the evolution of the agent's strategies is determined ``model.py``: contains the model level data including the position of all the agents and the agent ``schedule``. The ``datacollector`` collects data about the populations of each strategy.
* ``config.py`` is the interpreter of the json config files.
* ``logger.py`` provides the format for including logger statements as part of the code
* ``server.py`` runs the visualisation element of the program. Agents are represented with a percentage of their RBG colour while as the probabilities that they will play Rock, Paper or Scissors. The datacollector info is outputted as a chart.

## Further Reading

##### Prisoner's Dilemma

[Epstein, J. Zones of Cooperation in Demographic Prisoner's Dilemma. 1998.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.8629&rep=rep1&type=pdf)

##### Rock-Paper-Scissors

[Non-linear dynamics of rock-paper-scissors with Mutations](https://arxiv.org/pdf/1502.03370.pdf)

[Cyclic dominance in evolutionary games: a review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4191105/)