import numpy as np

# Função sigmoide e sua derivada
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Conjunto de dados de entrada
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Conjunto de dados de saída
y = np.array([[0, 0, 1, 1]]).T

# Semente para números aleatórios para tornar a execução determinística (boa prática)
np.random.seed(1)

# Inicializar pesos aleatoriamente com média 0
syn0 = 2 * np.random.random((3, 1)) - 1

# Treinamento
for iter in range(10000):  # Corrigido para usar range no Python 3

    # Propagação direta
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # Quanto erramos?
    l1_error = y - l1

    # Multiplicar quanto erramos pela derivada da sigmoide nos valores de l1
    l1_delta = l1_error * nonlin(l1, True)

    # Atualizar os pesos
    syn0 += np.dot(l0.T, l1_delta)

print("Saída após o treinamento:")
print(l1)
