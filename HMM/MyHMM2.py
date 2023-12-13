#!/usr/bin/env python3

# input: transition matrix A, emission(observation) matrix B, initial state probability PI
# output: the most probable sequence of states

def viterbi(A, B, PI, sequence):
    num_states = len(A)
    T = len(sequence)

    # Path probability matrix V
    Delta = [[0 for _ in range(num_states)] for _ in range(T)]
    # The path
    path = [0 for _ in range(T)]

    # Initialize
    for i in range(num_states):
        Delta[0][i] = PI[0][i] * B[i][sequence[0]]

    # For each t > 0
    for t in range(1, T):
        for i in range(num_states):
            max_prob = max(Delta[t-1][prev_s] * A[prev_s][i] * B[i]
                           [sequence[t]] for prev_s in range(num_states))
            Delta[t][i] = max_prob

    # Probability of best path
    # The final probability is the max probability at the last time step
    max_prob = max(Delta[T-1])
    path[T-1] = Delta[T-1].index(max_prob)

    # Backtrack to find the most probable path

    # Delta[t][s] * A[s][path[t+1]] = The product of the probability of the most likely path up to time t ending in state s
    # and the transition probability from state s to the state in the next time step
    for t in range(T-2, -1, -1):
        path[t] = max(range(num_states), key=lambda s: Delta[t]
                      [s] * A[s][path[t+1]])

    return path


count = 0
matrixs = []

while True:
    try:
        data = input()
        # print(data)
        if not data:
            break
        splitData = data.split()

        if (count < 3):
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

A = matrixs[0]   # transitionMatrix
B = matrixs[1]   # emissionMatrix
PI = matrixs[2]  # initialState


states = viterbi(A, B, PI, sequence)
for i in states:
    print(i, end=' ')
