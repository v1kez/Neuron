import numpy as np
from neuron import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('neuron_weights.txt')

# Пример использования
test_data = np.array([[18,0,0]])
predictions = new_neuron.forward(test_data)
print("Предсказанные значения:", predictions)
results = np.where(predictions >= 0.5, 'успех', 'провал')
print("Результаты:", results)


test_data = np.array([[30, 1,1]])
predictions = new_neuron.forward(test_data)
print("Предсказанные значения: ", predictions)
results = np.where(predictions >= 0.5, 'успех', 'провал')
print("Результаты:", results)