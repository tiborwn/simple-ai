import math

trainingData = [
    # temperature, sales (0 - low, 1 - high)

    (10, 0),
    (24, 1),
    (12, 0),
    (34, 1),
    (16, 0),
    (26, 1),
    (22, 0),
    (36, 1),

    (28, 1),
    (18, 0),
    (20, 0),
    (30, 1),
    (32, 1),
    (14, 0),
]

weight = 1.389294500177927
bias = -31.924045807944026
learningRate = 0.0001
trainingMode = False

def neuron(inputToken):
    dataComputed = weight * inputToken + bias
    predictionSigmoid = 1 / (1 + math.e ** -dataComputed)
    if predictionSigmoid > 0.5:
        return (1, predictionSigmoid)
    else:
        return (0, predictionSigmoid)
    
inputToken = int(input("Temperature: "))
result = neuron(inputToken)[0]
if result:
    print('high')
else: 
    print('low')

if trainingMode:
    avarageLoss = 0

    print("old")
    print(weight)
    print(bias)

    for temperature, correct in trainingData:
        output = neuron(temperature)
        loss = (output[1] - correct) ** 2
        avarageLoss += loss
        

    avarageLoss /= len(trainingData)

    print("Avarage loss before: ", avarageLoss)

    for i in range(0, 10000000):
        for temperature, correct in trainingData:
            output = neuron(temperature)
            loss = (output[1] - correct) ** 2
            bias = bias - (learningRate * ((2 * (output[1] - correct)) * (output[1] * (1 - output[1]))))
            weight = weight - (learningRate * (((2 * (output[1] - correct)) * (output[1] * (1 - output[1]))) * temperature)) 

    newAvarageLoss = 0

    for temperature, correct in trainingData:
        output = neuron(temperature)
        loss = (output[1] - correct) ** 2
        newAvarageLoss += loss

    newAvarageLoss /= len(trainingData)

    print("Avarage loss new: ", newAvarageLoss)

    if newAvarageLoss < avarageLoss:
        print("OK TRAINING")
    else:
        print("BAD TRAINING")

    print("new")
    print(weight)
    print(bias)

    result = neuron(1)
    if result:
        print('high')
    else: 
        print('low')

    for i in range (0, 30):
        print("TEST FOR ", i, ": ", neuron(i)[0])

