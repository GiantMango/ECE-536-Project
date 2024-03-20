import numpy as np
import cv2 as cv
import warnings

## Absolute mean brightness error (AMBE)
def ambe(input, output):
  x = np.mean(input)
  y = np.mean(output)
  return x - y


## Entropy: evaluate the efficiency of preserving details
def entropy(img):
  warnings.simplefilter('ignore', RuntimeWarning) # ignore runtimewarnings because log2(0) is handled
  h, w, ch = img.shape
  n = h* w
  hist = np.zeros((256, ch))
  for c in range(ch):
    hist[:,c] = cv.calcHist([img[:,:,c]], [0], None, [256], [0, 256]).flatten()
  pdf = hist / n
  pdf = np.round(np.average(pdf, axis=1), 3)
  a = np.nan_to_num(pdf*np.log2(pdf), nan = 0) # Handle log2(0) = 0
  e = -np.round(np.sum(a),3)
  return e


## Contrast
def contrast(img):
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten()
  cdf = np.cumsum(hist)
  quartile1_idx = np.int32(cdf[-1] * 0.25)
  quartile1 = np.sort(img.flatten())[quartile1_idx]
  quartile3_idx = np.int32(cdf[-1] * 0.75)
  quartile3 = np.sort(img.flatten())[quartile3_idx]
  # print("q1 i:", quartile1_idx, "q3 i:", quartile3_idx)
  # print("q1:", quartile1, "q3:", quartile3)
  hs = round((quartile3 - quartile1) / (np.max(img) - np.min(img)), 4)
  return hs
  

## Mean Square Error
def mse(input, output):
  return