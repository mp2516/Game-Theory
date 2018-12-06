# Game Theory Simulator

## Summary

This game-theory simulator can play a number of different games

### Rock-Paper-Scissors

#### Game Modes

* **Pure Only** pure strategies are only allowed. Worth varying:
    * The biome sizes
    * The probability of adoption and mutation
    * The initial population proportions
* **Pure and Perfect** pure strategies and perfect mixed are in play here.
* **Imperfect** only the imperfect_mixed strategies are present. Vary:
    * The strength of adoption
    * The probability of mutation

#### Strategies

* **pure_rock** always plays rock
* **pure_scissors** always plays scissors
* **pure_paper** always plays paper
* **perfect_mixed** chooses rock, paper or scissors with an equal probability
* **imperfect_mixed** chooses rock, paper or scissors with an unequal probability

### Prisoner's Dilemma

The Prisoner's Dilemma demonstrates how simple rules can lead to the emergence of widespread cooperation, despite the Defection strategy dominating each individual interaction game. However, it is also interesting for another reason: it is known to be sensitive to the activation regime employed in it.

#### Game Modes

* **All Strategies** implements all the below strategies against each other, tit_for_tat will usually win, but consider varying:
    * The initial population sizes

#### Strategies

* **all_c** always cooperates
* **all_d** always defects
* **tit_for_tat** cooperates on the first move then plays what its opponent played the previous move (Rapoport & Chammah 1965).
* **spiteful** cooperates until the opponent defects and thereafter always defects (Axelrod 2006). Sometimes also called grim.
* **random** chooses cooperate or defect at random.


## General Concepts

* _biomes_. Biomes are areas of homogenous strategies, they are squares on the grid and are dictated by the _biome_size_, an attribute of the model module.


## Files

* ``run.py`` is the entry point for the front-end simulations.
* ``agent.py``: contains the agent class which dictates how the agents play each other. Here the strategies of the agents are determined and the evolution of the agent's strategies is determined ``model.py``: contains the model level data including the position of all the agents and the agent ``schedule``. The ``datacollector`` collects data about the populations of each strategy.
* ``config.py`` is the interpreter of the json config files.
* ``logger.py`` provides the format for including logger statements as part of the code
* ``server.py`` runs the visualisation element of the program. Agents are represented with a percentage of their RBG colour while as the probabilities that they will play Rock, Paper or Scissors. The datacollector info is outputted as a chart.

## Further Reading

### Prisoner's Dilemma

[Epstein, J. Zones of Cooperation in Demographic Prisoner's Dilemma. 1998.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.8629&rep=rep1&type=pdf)

[Optimal Strategies of the Iterated Prisoner’s Dilemma Problem for Multiple Conflicting Objectives](https://www.iitk.ac.in/kangal/papers/k2006002.pdf)

### Rock-Paper-Scissors

[Non-linear dynamics of rock-paper-scissors with Mutations](https://arxiv.org/pdf/1502.03370.pdf)

[Cyclic dominance in evolutionary games: a review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4191105/)

[Oscillitory dynamics of Rock-Paper-Scissors](https://ac.els-cdn.com/S0022519310000123/1-s2.0-S0022519310000123-main.pdf?_tid=8a0c7e4f-10b1-40b8-96df-ee9aa9ab4a05&acdnat=1543783308_9aa411b2def1624726fc1b60352bcb9e)

A textbook like paper describing the fundementals of evolutionary games, cited as the founding paper of the field.
[Evolutionary games and population Dynamics](http://baloun.entu.cas.cz/krivan/papers/kamenice13.pdf)

[Mobility promotes and jeopardises biodiversity in rock-paper scissors game](https://www.nature.com/articles/nature06095.pdf)

1994, a fundemental paper that started the field off
[Vortices and Strings in Model ecosystems](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.50.3401)