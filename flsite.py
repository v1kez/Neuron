import pickle

import numpy as np
from flask import Flask, render_template, url_for, request, jsonify
from model.neuron import SingleNeuron

app = Flask(__name__)

menu = [
        {"name": "Neuron", "url": "p_lab4"}]


# Загрузка весов из файла
new_neuron = SingleNeuron(input_size=2)
new_neuron.load_weights('model/neuron_weights.txt')

@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные Буренок Д.Р.", menu=menu)



@app.route("/p_lab4", methods=['POST', 'GET'])
def p_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Первый нейрон", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        predictions = new_neuron.forward(X_new)
        print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Состоит в браке', 'Не состоит в браке'))
        return render_template('lab4.html', title="Первый нейрон", menu=menu,
                               class_model="Он(а) " + str(*np.where(predictions >= 0.5, 'Состоит в браке', 'Не состоит в браке')))



if __name__ == "__main__":
    app.run(debug=True)
