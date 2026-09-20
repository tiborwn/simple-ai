# Simple AI models examples
Written for studying AI and ML

### icecream-sales
>#### Provide the temperature and get sales amount (low / high).
>- **1 input neuron**
>- **1 output neuron**
>- Output activation: Sigmoid
>- Loss: Mean Squared Error
>- Optimizer: Stochastic Gradient Descent (batch size = 1)

### comfortable-temperature
>#### Provide the temperature and find out how comfortable is it (uncomfortable / comfortable).
>- **1 input neuron**
>- **2 hidden neurons**
>- **1 output neuron**
>- Hidden activation: ReLU
>- Output activation: Sigmoid
>- Loss: Mean Squared Error
>- Optimizer: Stochastic Gradient Descent (batch size = 1)

### shape-classifier
>#### Classify a 8x8 image as a vertical line, horizontal line, plus, or X..
>- **64 input neurons**
>- **16 neurons in first hidden layer**
>- **8 neurons in second hidden layer**
>- **4 output neurons**
>- Hidden activation: ReLU
>- Output activation: Softmax
>- Loss: Cross-Entropy Error
>- Optimizer: Stochastic Gradient Descent (batch size = 1)