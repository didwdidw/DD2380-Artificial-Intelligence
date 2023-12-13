#!/usr/bin/env python3

# input: transition matrix A, emission(observation) matrix B, initial state probability PI
# output: probability of given sequence

def forward_algo(A, B, PI, sequence):
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


probability = forward_algo(A, B, PI, sequence)
print(round(probability, 6))
