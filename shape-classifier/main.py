from PIL import Image
import numpy as np

inputImage = Image.open('input.png')
print(np.array(inputImage))