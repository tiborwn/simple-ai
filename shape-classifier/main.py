from PIL import Image
import numpy as np

inputImage = Image.open('input.png').convert('RGB')
inputImageTokens = []
for line in np.array(inputImage):
    for i in range(0, len(line)):
        if np.all(line[i] == [255, 255, 255]):
            inputImageTokens.append(1)
        else:
            inputImageTokens.append(0)
