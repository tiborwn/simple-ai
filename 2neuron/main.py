import json, math

parameters = json.load(open('parameters.json'))

trainingData = [
    # temperature, comfortable (0 - uncomfortable, 1 - comfortable)

    (28, 0),
    (5, 0),
    (18, 1),
    (10, 0),

    (15, 1),
    (32, 0),
    (20, 1),
    (0, 0),
    (30, 0),
    (22, 1),
    (40, 0),
    (12, 0),

    (35, 0),
    (25, 1),
]

learningRate = 0.01

def neuron(i, inputToken):
    neuronWeight = parameters['hiddenLayer'][i]['weight']
    neuronBias = parameters['hiddenLayer'][i]['bias']
    computed = neuronWeight * inputToken + neuronBias
    return max(0, computed)

def output(inputToken):
    outputToken = 0
    outputBias = parameters['outputLayer']['bias']
    for i in range(0, 2):
        relu = neuron(i, inputToken)
        outputWeight = parameters['outputLayer']['weights'][i]
        outputToken += outputWeight * relu
    outputToken += outputBias
    predictionSigmoid = 1 / (1 + math.e ** -outputToken)
    return predictionSigmoid

def calculateLoss(trainingData):
    averageLoss = 0
    for temperature, target in trainingData:
        averageLoss += (output(temperature) - target) ** 2
    averageLoss /= len(trainingData)
    return averageLoss

def reluDerivative(num):
    return 1 if num > 0 else 0

trainingMode = -1

while trainingMode not in range(0, 2):
    trainingMode = int(input("Training? 0 or 1: "))

if trainingMode:
    lossBefore = calculateLoss(trainingData)
    print("LOSS BEFORE: ", lossBefore)

    epoches = int(input("Epoches: "))
    
    for i in range(0, epoches):
        for temperature, target in trainingData:
            outputToken = 0
            outputBias = parameters['outputLayer']['bias']

            for i in range(0, 2):
                relu = neuron(i, temperature)
                outputWeight = parameters['outputLayer']['weights'][i]
                outputToken += outputWeight * relu
            outputToken += outputBias

            predictionSigmoid = 1 / (1 + math.e ** -outputToken)
            deltaOut = 2 * (predictionSigmoid - target) * (predictionSigmoid * (1 - predictionSigmoid))

            for i in range(0, 2):
                outputWeight = parameters['outputLayer']['weights'][i]
                weight = parameters['hiddenLayer'][i]['weight']
                bias = parameters['hiddenLayer'][i]['bias']
                raw = weight * temperature + bias
                relu = neuron(i, temperature)


                weightGradient = (deltaOut * outputWeight * reluDerivative(raw) * temperature)
                biasGradient = (deltaOut * outputWeight * reluDerivative(raw))

                parameters['hiddenLayer'][i]['weight'] = parameters['hiddenLayer'][i]['weight'] - (learningRate * weightGradient)
                parameters['hiddenLayer'][i]['bias'] = parameters['hiddenLayer'][i]['bias'] - (learningRate * biasGradient)
                parameters['outputLayer']['weights'][i] = parameters['outputLayer']['weights'][i] - (learningRate * (deltaOut * relu))

            parameters['outputLayer']['bias'] = parameters['outputLayer']['bias'] - (learningRate * deltaOut)

    lossAfter = calculateLoss(trainingData)
    print("LOSS AFTER: ", lossAfter)

    if lossAfter < lossBefore:
        print("OK TRAINING")
        with open("parameters.json", "w", encoding="utf-8") as file:
            json.dump(parameters, file, indent=2, ensure_ascii=False)
    else:
        print("WRONG TRAINING")
else: 
    if output(int(input('Temperature: '))) > 0.5:
        print('comfortable')
    else: 
        print('uncomfortable')