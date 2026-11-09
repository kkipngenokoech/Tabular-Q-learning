# Tabular Q-learning with a charger grid

A 45-minute introduction to tabular Q-learning for readers who know basic Python. The agent starts at A and learns to reach the charger at D.

| | Left | Right |
|---|---|---|
| Top | A (start) | B |
| Bottom | C | D (charger) |

Entering D earns +10 and ends the episode. Every other move earns 0. A move into a wall leaves the agent in place.

## Materials

- [Learning plan](Tabular_Q_Learning_Plan.docx)
- [Slides](Tabular_Q_Learning_Slides.pptx)
- [Custom Gymnasium environment and training script](charger_grid.py)

## Run the example

```sh
python -m pip install gymnasium
python charger_grid.py --episodes 6
```

The script prints the route, total reward and Q-table after every episode. The first two episodes replay Right then Down so you can check the values by hand. From episode 3, the agent chooses actions using epsilon-greedy exploration.

| After | Q(A, Right) | Q(B, Down) |
|---|---:|---:|
| Initialization | 0 | 0 |
| Episode 1 | 0 | 5 |
| Episode 2 | 2.25 | 7.5 |

Run again with `--episodes 20 --epsilon 0.5` and compare which routes the agent tries and when values in row C change.
