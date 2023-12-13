#!/usr/bin/env python3

import math
import time

# input: transition matrix A, emission(observation) matrix B, initial state probability PI
# output: estimated transition matrix and emission matrix


def forward_algo(A, B, PI, sequence):
    N = len(A)
    T = len(sequence)

    Alpha = [[0 for _ in range(N)] for _ in range(T)]
    # The sum of the probabilities of transitioning from all possible states to state i at time t.
    # It is used to normalize the forward probabilities.
    scaling_factors = [0 for _ in range(T)]

    # Initialize
    scaling_factors[0] = sum(PI[0][i] * B[i][sequence[0]] for i in range(N))
    for i in range(N):
        Alpha[0][i] = PI[0][i] * B[i][sequence[0]] / scaling_factors[0]

    # For 1 <= t <= T-1
    for t in range(1, T):
        scaling_factors[t] = sum(Alpha[t-1][j] * A[j][i]
                                 for i in range(N) for j in range(N))
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
                A[i][j] = numerator / denominator

        # Update B
        for j in range(N):
            for k in range(K):
                numerator = sum(Gamma[t][j]
                                for t in range(T-1) if sequence[t] == k)
                denominator = sum(Gamma[t][j] for t in range(T-1))
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
        if elapsed_time > 0.9:
            break

    return A, B


def format_matrix(matrix):
    formatted_matrix = []
    for row in matrix:
        formatted_row = ["{:.6g}".format(num) if isinstance(
            num, float) else num for num in row]
        formatted_matrix.append(formatted_row)
    return formatted_matrix


start_time = time.time()
count = 0
matrixs = []

while True:
    try:
        data = input()
        if not data:
            break
        splitData = data.split()

        if count < 3:
            count += 1
            rows = int(splitData[0])
            cols = int(splitData[1])
            values = [float(i) for i in splitData[2:]]
            matrix = [values[i:i+cols] for i in range(0, len(values), cols)]
            matrixs.append(matrix)
        else:
            splitData = data.split()
            num_of_seq = int(splitData[0])
            sequence = [int(i) for i in splitData[1:]]
    except EOFError:
        break

A = matrixs[0]
B = matrixs[1]
PI = matrixs[2]

estimated_A, estimated_B = baum_welch(A, B, PI, sequence, 100, start_time)

print(len(estimated_A), end=' ')
print(len(estimated_A[0]), end=' ')
for i in format_matrix(estimated_A):
    for j in i:
        print(j, end=' ')
print("")

print(len(estimated_B), end=' ')
print(len(estimated_B[0]), end=' ')
for i in format_matrix(estimated_B):
    for j in i:
        print(j, end=' ')
