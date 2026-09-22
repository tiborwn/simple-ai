import math
import random
import json

parameters = {}

parameters['hiddenLayer1'] = []
parameters['hiddenLayer2'] = []
parameters['outputLayer'] = []

for i in range (0, 16):
    weights = []
    for k in range(0, 64):
        randomWeight = random.uniform(-math.sqrt(2 / 64), math.sqrt(2 / 64))
        weights.append(randomWeight)

    neuron = {
        'weights': weights,
        'bias': 0
    }

    parameters['hiddenLayer1'].append(neuron)

for i in range (0, 8):
    weights = []
    for k in range(0, 16):
        randomWeight = random.uniform(-math.sqrt(2 / 16), math.sqrt(2 / 16))
        weights.append(randomWeight)

    neuron = {
        'weights': weights,
        'bias': 0
    }
    
    parameters['hiddenLayer2'].append(neuron)

for i in range (0, 4):
    weights = []
    for k in range(0, 8):
        randomWeight = random.uniform(-math.sqrt(2 / 8), math.sqrt(2 / 8))
        weights.append(randomWeight)

    neuron = {
        'weights': weights,
        'bias': 0
    }
    
    parameters['outputLayer'].append(neuron)

with open('parameters.json', mode='w') as file:
    json.dump(parameters, file, indent=2)