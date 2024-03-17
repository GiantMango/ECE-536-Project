import numpy as np
import cv2 as cv

## Absolute mean brightness error (AMBE)
def ambe(input, output):
  x = np.average(input.flatten())
  y = np.average(output.flatten())
  return x - y


## Entropy: evaluate the efficiency of preserving details
def entropy(input, output):
  
  hist = cv.calcHist([input], [0], None, [256], [0, 256]).flatten()
  pdf = hist / len()