import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import assessment

## Daily Pictures (ExDark)
# https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
# 
## Underwater Dataset (EUVP)
# https://irvlab.cs.umn.edu/resources/euvp-dataset


def main():
  try:
    path = "images/"+sys.argv[1]
  except:
    path = "images/15045.png"
  
  
  if len(sys.argv) == 2: ## Gray-Scaled
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    qdhe_img = qdhe(img)
    
  elif len(sys.argv) == 3: ## RGB
    img = cv.imread(path)
    b,g,r = cv.split(img)
    
    ## QDHE
    qdhe_r = qdhe(r, display='off')
    qdhe_g = qdhe(g, display='off')
    qdhe_b = qdhe(b, display='off')
    qdhe_img = cv.merge((qdhe_b, qdhe_g, qdhe_r))

    ## Conventional Histogram Equilization    
    he_r = cv.equalizeHist(r)
    he_g = cv.equalizeHist(g)
    he_b = cv.equalizeHist(b)
    he_img = cv.merge((he_b, he_g, he_r))
    
    ## CLANE
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_r = clahe.apply(r)
    clahe_g = clahe.apply(g)
    clahe_b = clahe.apply(b)
    clahe_img = cv.merge((clahe_b, clahe_g, clahe_r))
    
    
    plt.figure(figsize=(8, 6))
    displayImage(img, qdhe_img)
    plt.show()

  return qdhe_img


## plot [0 255] histogram from image
def pltHist(im):
  plt.bar(np.arange(256), cv.calcHist([im.astype(np.uint8)], [0], None, [256], [0, 256]).flatten())
  
def displayImage(input, output, he_img=None, clahe_img=None):
  if len(sys.argv) == 3:
    print("RGB Mode")
    cmap = None
    input = cv.cvtColor(input.astype(np.uint8), cv.COLOR_BGR2RGB)
    output = cv.cvtColor(output.astype(np.uint8), cv.COLOR_BGR2RGB)
    if he_img is not None:
      he_img = cv.cvtColor(he_img.astype(np.uint8), cv.COLOR_BGR2RGB)
    if clahe_img is not None:
      clahe_img = cv.cvtColor(clahe_img.astype(np.uint8), cv.COLOR_BGR2RGB)
  else:
    cmap ='gray'
    
  plt.subplot(2, 4, 1)
  plt.imshow(input, cmap=cmap)
  plt.title('Original Image')
  plt.axis('off')
  plt.subplot(2, 4, 5)
  pltHist(input)
  plt.title('Input Image Histogram')

  plt.subplot(2, 4, 2)
  plt.imshow(output, vmin=0, vmax=255)
  plt.title('QDHE Image')
  plt.axis('off')
  plt.subplot(2, 4, 6)
  pltHist(output)
  plt.title('QDHE Histogram')
  
  if he_img != None:
    plt.subplot(2, 4, 3)
    plt.imshow(he_img, vmin=0, vmax=255)
    plt.title('HE Image')
    plt.axis('off')
    plt.subplot(2, 4, 7)
    pltHist(he_img)
    plt.title('HE Histogram')
  
  if clahe_img != None:
    plt.subplot(2, 4, 4)
    plt.imshow(clahe_img, vmin=0, vmax=255)
    plt.title('CLAHE Image')
    plt.axis('off')
    plt.subplot(2, 4, 8)
    pltHist(clahe_img)
    plt.title('CLAHE Histogram')

  plt.tight_layout()


## Input one channel of image, either gray, red, green, or blue
def qdhe(img, display='on', debug = False):
  hist = cv.calcHist([img], [0], None, [256], [0, 256]).flatten().astype(int)
  n = np.sum(hist, dtype=int)
  img_array = np.sort(img.flatten())
  
  ## Sub histogram partition
  m = np.zeros(5, dtype=int)
  m[0] = img_array[0]
  m[1] = img_array[n//4]
  m[2] = img_array[n//2]
  m[3] = img_array[n*3//4]
  m[4] = img_array[-1]

  ## Clipping
  tc = n // 256
  clipped_hist = np.minimum(hist, tc)
    
  ## Calculate histogram equalization mapping
  span = np.diff(m)
  range_ = (255 * span // sum(span)).astype(int)
  new_m = np.cumsum(np.insert(range_, 0, 0))
  new_m[-1] = 255
  normalized_cdf = np.zeros(256)
  map_ = np.zeros(256, dtype=int)
 
  ## Allocate clipped data
  for i in range(4):
    clipped_cdf = np.cumsum(clipped_hist[m[i]:m[i+1]])
    normalized_cdf[m[i]:m[i+1]] = clipped_cdf / clipped_cdf[-1]
    map_[m[i]:m[i+1]] = range_[i] * normalized_cdf[m[i]:m[i+1]] + new_m[i]

  # Fix overexposure spots
  max_idx = np.argmax(map_)
  map_[max_idx:] = 255

  # Histogram equalization
  QDHE_img = map_[img]
  
  
  ## DEBUG
  if debug:
    print("Thresholds:", m)  
    print("span:", span)
    print("range:", range_)
    print("New Thresholds:", new_m)
  
  
  ## Plot image
  if display == 'on':
    plt.figure(figsize=(8, 6))

    displayImage(img, QDHE_img)
    
    ## Plot dashed line on seperate points
    plt.subplot(2,4,5)
    for i in range(len(m)):
      plt.axvline(x = m[i], linestyle='--', color="r", lw = "0.5")
    plt.subplot(2,4,6)
    for i in range(len(new_m)):
      plt.axvline(x = new_m[i], linestyle='--', color="r", lw = "0.5")
    plt.axhline(y = tc, linestyle='--', color="b", lw = "0.5")
    plt.show()    
    
    plt.step(np.arange(256), map_)
    plt.show()
        
  return QDHE_img

if __name__ == "__main__":
  main()  