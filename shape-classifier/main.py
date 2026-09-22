from PIL import Image
import json
import math
import numpy as np

with open('parameters.json', mode='r') as file:
    parameters = json.load(file)

inputImage = Image.open('input.png').convert('RGB')
inputImageTokens = []
for line in np.array(inputImage):
    for i in range(0, len(line)):
        if np.all(line[i] == [255, 255, 255]):
            inputImageTokens.append(1)
        else:
            inputImageTokens.append(0)

def neuron(layerIndex, neuronIndex, inputTokens):
    neuronOutput = 0

    for k, v in enumerate(inputTokens):
        neuronOutput += parameters[layerIndex][neuronIndex]['weights'][k] * v

    neuronOutput += parameters[layerIndex][neuronIndex]['bias']

    return max(0, neuronOutput) if layerIndex != 2 else neuronOutput

def softmax(numbers):
    result = []
    numerators = []
    denominator = 0

    biggest = 0

    for i in numbers:
        if biggest < i:
            biggest = i

    for i in numbers:
        numerators.append(np.e ** (i - biggest))
        denominator += np.e ** (i - biggest)

    for i in numerators:
        result.append(i / denominator)

    return result

def output():
    firstLayerOutput = []
    secondLayerOutput = []    
    rawOutput = []

    for n in range(0, 16):
        firstLayerOutput.append(neuron(0, n, inputImageTokens))

    for n in range(0, 8):
        secondLayerOutput.append(neuron(1, n, firstLayerOutput))

    for o in range(0, 4):
        rawOutput.append(neuron(2, o, secondLayerOutput))

    return softmax(rawOutput)

biggestProbability = 0
modelOutput = output()

for p in modelOutput:
    if biggestProbability < p:
        biggestProbability = p

match modelOutput.index(biggestProbability):
    case 0:
        print('vertical line')
    case 1:
        print('horizontal line')
    case 2:
        print('plus')
    case 3:
        print('cross')