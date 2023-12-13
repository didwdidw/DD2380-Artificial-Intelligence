#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import math
import numpy as np
import random
import time
import sys

epsilon = sys.float_info.epsilon


def predict_forward(A, B, PI, sequence):
    # calculate probability
    # forward_matrix, size  =  length of sequence  x  #states
    Alpha = [[0 for _ in range(len(A))] for _ in range(len(sequence))]

    # Initialize
    for i in range(len(A)):
        Alpha[0][i] = PI[0][i] * B[i][sequence[0]]

    # For 1 <= t <= T-1
    for t in range(1, len(sequence)):
        for i in range(len(A)):
            sum_prob = 0
            for j in range(len(A)):
                sum_prob += Alpha[t-1][j] * A[j][i]
            Alpha[t][i] = sum_prob * B[i][sequence[t]]

    total_prob = sum(Alpha[-1])
    return total_prob


def forward_algo(A, B, PI, sequence):
    N = len(A)
    T = len(sequence)

    Alpha = [[0 for _ in range(N)] for _ in range(T)]
    # The sum of the probabilities of transitioning from all possible states to state i at time t.
    # It is used to normalize the forward probabilities.
    scaling_factors = [0 for _ in range(T)]

    # Initialize
    scaling_factors[0] = sum(PI[0][i] * B[i][sequence[0]] for i in range(N))
    scaling_factors[0] += epsilon  # Make sure it is never zero
    for i in range(N):
        Alpha[0][i] = PI[0][i] * B[i][sequence[0]] / scaling_factors[0]

    # For 1 <= t <= T-1
    for t in range(1, T):
        scaling_factors[t] = sum(Alpha[t-1][j] * A[j][i]
                                 for i in range(N) for j in range(N))
        scaling_factors[t] += epsilon  # Make sure it is never zero
        for i in range(N):
            Alpha[t][i] = sum(Alpha[t-1][j] * A[j][i]
                              for j in range(N)) * B[i][sequence[t]] / scaling_factors[t]

    return Alpha, scaling_factors


def backward_algo(A, B, sequence, scaling_factors):
    # Use the scaling_factors from the calculation of the forward algorithm.
    # This ensures consistency in scaling between the forward and backward probabilities.
    N = len(A)
    T = len(sequence)

    Beta = [[0 for _ in range(N)] for _ in range(T)]

    # Initialize
    for i in range(N):
        Beta[T - 1][i] = 1 / scaling_factors[T-1]

    # For t < T-1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            Beta[t][i] = sum(A[i][j] * B[j][sequence[t + 1]] * Beta[t + 1][j]
                             for j in range(N)) / scaling_factors[t]

    return Beta


def baum_welch(A, B, PI, sequence, max_iters, start_time):
    N = len(A)         # Number of possible states
    K = len(B[0])      # Number of possible observations
    T = len(sequence)  # Length of the given sequence
    # print(N, file=sys.stderr)
    # print(K, file=sys.stderr)
    # print(T, file=sys.stderr)

    oldLogProb = float('-inf')

    for iters in range(max_iters):
        # Scaling probabilities in the context of the Baum-Welch algorithm does not affect the correctness of the computations.
        # This is because the scaling is applied uniformly to all probabilities at a particular time step,
        # maintaining the relative relationships between probabilities.
        Alpha, scaling_factors = forward_algo(A, B, PI, sequence)
        Beta = backward_algo(A, B, sequence, scaling_factors)

        Di_Gamma = [[[0 for _ in range(N)] for _ in range(N)]
                    for _ in range(T - 1)]
        for t in range(T-1):
            for i in range(N):
                for j in range(N):
                    numerator = Alpha[t][i] * A[i][j] * \
                        B[j][sequence[t+1]] * Beta[t+1][j]
                    denominator = sum(Alpha[T-1][k] for k in range(N))
                    if denominator == 0:
                        denominator = epsilon
                    Di_Gamma[t][i][j] = numerator / denominator

        Gamma = [[0 for _ in range(N)] for _ in range(T-1)]
        for t in range(T-1):
            for i in range(N):
                Gamma[t][i] = sum(Di_Gamma[t][i][j] for j in range(N))

        # Update A
        for i in range(N):
            for j in range(N):
                numerator = sum(Di_Gamma[t][i][j] for t in range(T-1))
                denominator = sum(Gamma[t][i] for t in range(T-1))
                if denominator == 0:
                    denominator = epsilon
                A[i][j] = numerator / denominator

        # Update B
        for j in range(N):
            for k in range(K):
                numerator = sum(Gamma[t][j]
                                for t in range(T-1) if sequence[t] == k)
                denominator = sum(Gamma[t][j] for t in range(T-1))
                if denominator == 0:
                    denominator = epsilon
                B[j][k] = numerator / denominator

        # Update PI
        for i in range(N):
            PI[0][i] = Gamma[0][i]

        # The algorithm stops when the change in log-likelihood between iterations is negligible.
        # This indicates that the algorithm has converged to a stable solution, further iterations are unlikely to significantly improve the model.
        logProb = -sum(math.log(1 / scaling_factors[t]) for t in range(T))
        if logProb <= oldLogProb:
            break
        oldLogProb = logProb

        elapsed_time = time.time() - start_time
        if elapsed_time > 4.5:
            break

    return A, B, PI


def random_row_stochastic_generator(size):
    random_list = [(1 / size) + np.random.rand() / 1000 for _ in range(size)]
    return [element / sum(random_list) for element in random_list]


class Lambda:
    def __init__(self, species, emissions):
        self.A = [random_row_stochastic_generator(
            species) for _ in range(species)]  # size = N * N
        self.B = [random_row_stochastic_generator(
            emissions) for _ in range(species)]  # size = N * K
        self.PI = [random_row_stochastic_generator(species)]  # size = 1 * N

    def set_A(self, A):
        self.A = A

    def set_B(self, B):
        self.B = B

    def set_PI(self, PI):
        self.PI = PI


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        # N_FISH * (fish id, [observations])
        self.fishes_observation = [(i, []) for i in range(N_FISH)]
        self.fishes_lambda = [Lambda(1, N_EMISSIONS)
                              for _ in range(N_SPECIES)]  # size = N_SPECIES * Lambda

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        self.start_time = time.time()

        # Record the observation of each fish
        # len(self.fishes_observation) will decrease overtime
        for i in range(len(self.fishes_observation)):
            self.fishes_observation[i][1].append(observations[i])

        # Only start to guess at the last 70 timesteps
        if step < N_STEPS - N_FISH:
            return None

        # Start guessing
        fish_id, observation = self.fishes_observation.pop()
        fish_specie = 0
        max_prob = 0
        for l, s in zip(self.fishes_lambda, range(N_SPECIES)):
            prob = predict_forward(l.A, l.B, l.PI, observation)
            if prob > max_prob:
                max_prob = prob
                fish_specie = s
        self.observation = observation
        return fish_id, fish_specie

    def update_lambda(self, fish_specie):
        # Record the pattern of each specie
        A, B, PI = baum_welch(self.fishes_lambda[fish_specie].A, self.fishes_lambda[fish_specie].B,
                              self.fishes_lambda[fish_specie].PI, self.observation, 1000, self.start_time)
        self.fishes_lambda[fish_specie].set_A(A)
        self.fishes_lambda[fish_specie].set_B(B)
        self.fishes_lambda[fish_specie].set_PI(PI)

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        if not correct:
            self.update_lambda(true_type)
