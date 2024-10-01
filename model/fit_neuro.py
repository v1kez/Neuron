import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
X = np.array([[20, 0, 0],
  [25, 1, 1],
  [30, 0, 1],
  [19, 1, 0],
# [20, 0, 0],
#   [25, 1, 1],
#   [17, 1, 0],
#   [28, 1, 1],
#   [19, 1, 0],
#   [30, 0, 1],
#   [31, 1, 1],
#   [18, 1, 1],
  ])
y = np.array([0,1,1,0])  # Ожидаемый выход
# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=3)
neuron.train(X, y, epochs=5000, learning_rate=0.01)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')