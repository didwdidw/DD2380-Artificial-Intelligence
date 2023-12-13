#!/usr/bin/env python3

# input: transition matrix A, emission(observation) matrix B, initial state probability PI
# output: emission(observation) probability B

matrixs = []

while True:
    try:
        data = input()
        # print(data)
        if not data:
            break

        splitData = data.split()

        rows = int(splitData[0])
        cols = int(splitData[1])

        values = [float(i) for i in splitData[2:]]

        matrix = [values[i:i+cols] for i in range(0, len(values), cols)]

        matrixs.append(matrix)
    except EOFError:
        break

A = matrixs[0]   # transitionMatrix
B = matrixs[1]   # emissionMatrix
PI = matrixs[2]  # initialState


# next_state = PI * A
next_state = [[0 for _ in range(len(A[0]))] for _ in range(len(PI))]
for i in range(len(PI)):
    for j in range(len(A[0])):
        for k in range(len(PI[0])):
            next_state[i][j] += PI[i][k] * A[k][j]

# next_emission = next_state * B
next_emission = [[0 for _ in range(len(B[0]))] for _ in range(len(next_state))]
for i in range(len(next_state)):
    for j in range(len(B[0])):
        for k in range(len(next_state[0])):
            next_emission[i][j] += next_state[i][k] * B[k][j]

print(len(next_emission), end=' ')
print(len(next_emission[0]), end=' ')
for i in next_emission[0]:
    print(i, end=' ')
