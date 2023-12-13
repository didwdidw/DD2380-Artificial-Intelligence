# Artificial Intelligence
We implement various AI algorithms for different problem scenarios.

## Search
Our approach involves the utilization of iterative deepening coupled with alpha-beta pruning to determine the optimal move for a given state. Additionally, we incorporate state storing for efficient repeated state checking, significantly enhancing overall performance.

## HMM
We delve into Hidden Markov Models and associated algorithms to predict desired information based on given data. In the fishing derby application, we leverage the Baum-Welch Algorithm to analyze observation sequences corresponding to various fish. This algorithm aids in identifying patterns unique to each fish species within the given data. Subsequently, we employ the Forward Algorithm on individual fish sequences, enabling us to predict the species to which each fish is most likely to belong.

## RL
We implement the Q-learning algorithm, employing a linear scheduling epsilon greedy policy for training. We thoroughly explore the hyperparameters of the model and analyze their impact on performance to optimize the learning process.
