import json, math

parameters = json.load(open('parameters.json'))

trainingData = [
    # temperature, rain, target (0 - dont run, 1 - run)

    (32, 0, 0),
    (2, 0, 0),
    (10, 5, 0),
    (24, 1, 1),
    (17, 7, 0),
    (14, 0, 1),
    (16, 1, 1),
    (20, 2, 1),
    (22, 0, 1),
    (8, 0, 0),
    (21, 9, 0),
    (26, 6, 0),
    (18, 0, 1),
    (5, 1, 0),
    (28, 8, 0),
    (30, 2, 0),
    (23, 5, 0),
    (35, 0, 0),
]

learningRate = 0.01

# 2 tokens in 
def neuron(i, inputTokens):
    neuronWeights = parameters['hiddenLayer'][i]['weights']
    neuronBias = parameters['hiddenLayer'][i]['bias']
    computed = 0
    for i in range(0, 2):
        computed += (neuronWeights[i] * inputTokens[i])
    computed += neuronBias
    return max(0, computed)

# 3 tokens in
def output(inputTokens):
    outputToken = 0
    outputBias = parameters['outputLayer']['bias']
    for i in range(0, 3):
        relu = neuron(i, inputTokens)
        outputWeight = parameters['outputLayer']['weights'][i]
        outputToken += outputWeight * relu
    outputToken += outputBias
    predictionSigmoid = 1 / (1 + math.e ** -outputToken)
    return predictionSigmoid

def calculateLoss(trainingData):
    averageLoss = 0
    for temperature, rain, target in trainingData:
        averageLoss += (output([temperature, rain]) - target) ** 2
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
        for temperature, rain, target in trainingData:
            inputTokens = [temperature, rain]
            outputToken = 0
            outputBias = parameters['outputLayer']['bias']

            for i in range(0, 3):
                relu = neuron(i, [temperature, rain])
                outputWeight = parameters['outputLayer']['weights'][i]
                outputToken += outputWeight * relu
            outputToken += outputBias

            predictionSigmoid = 1 / (1 + math.e ** -outputToken)
            deltaOut = 2 * (predictionSigmoid - target) * (predictionSigmoid * (1 - predictionSigmoid))

            for i in range(0, 3):
                outputWeight = parameters['outputLayer']['weights'][i]

                raw = 0
                for k in range(0, 2):
                    weight = parameters['hiddenLayer'][i]['weights'][k]
                    raw += weight * inputTokens[k]
                bias = parameters['hiddenLayer'][i]['bias']
                raw += bias

                relu = neuron(i, [temperature, rain])

                biasGradient = (deltaOut * outputWeight * reluDerivative(raw))

                for k in range(0, 2):
                    weightGradient = (deltaOut * outputWeight * reluDerivative(raw) * inputTokens[k])
                    parameters['hiddenLayer'][i]['weights'][k] = parameters['hiddenLayer'][i]['weights'][k] - (learningRate * weightGradient)
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
    temperature = int(input('Temperature: '))
    rain = int(input('Rain: '))
    prediction = output([temperature, rain])
    if prediction > 0.5:
        print('run')
    else: 
        print('dont run')