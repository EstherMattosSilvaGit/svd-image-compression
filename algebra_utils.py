import math

def matmul(A, B):
    """Multiplica duas matrizes A e B manualmente"""
    linhasA, colunasA = len(A), len(A[0])
    linhasB, colunasB = len(B), len(B[0])

    assert colunasA == linhasB, "Dimensões incompatíveis para multiplicação"

    C = [[0 for _ in range(colunasB)] for _ in range(linhasA)]

    for i in range(linhasA):
        for j in range(linhasB):
            for k in range(colunasA):
                C[i][j] += A[i][k] * B[k][j]
        
        return C
