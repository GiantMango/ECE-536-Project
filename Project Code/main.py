import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import assessment as assess
import nibabel as nib

## Daily Pictures (ExDark)
# https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
# 
## Underwater Dataset (EUVP)
# https://irvlab.cs.umn.edu/resources/euvp-dataset
#
## UIEB Challenging Dataset
#
#
## Medical Images from MedPix
# https://medpix.nlm.nih.gov/case?id=e6a1b9d9-3a26-463d-8764-b09a90186466

def main():
  ## Read Images
  try:
    path = "images/"+sys.argv[1]
    dc, filetype = path.split('.')
    img = cv.imread(path)
  except:
    path = "images/miplab-ncct_sym_brain.nii"
  
  ## Normal Image Filetypes
  if filetype in ["jpg", "jpeg", "png"]:
    if img.ndim == 2 or sys.argv[2] == 'gray': # Gray Scale
      print("------- GRAY MODE -------")
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      img = img[:,:,np.newaxis]
      ch = 1      
    elif img.ndim == 3 or sys.argv[2] == 'rgb': ## RGB
      print("------- RGB MODE -------")
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      ch = 3

  ## Medical Image Filetype
  elif filetype == "nii": # Medical
    print("------- Medical Image .nii --------")
    med_images = nib.load(path).get_fdata()
    h, w, num = med_images.shape
    img = med_images[:,:,num//2]
    img = img[:,:,np.newaxis]
    ch = 1
    plt.imshow(img)
    plt.show()
    

  ## Allocate memory for images
  qdhe_img = np.zeros_like(img, dtype=np.uint8)
  he_img = np.zeros_like(img, dtype=np.uint8)
  clahe_img = np.zeros_like(img, dtype=np.uint8)
  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  
  ## QDHE
  for i in range(ch):
    qdhe_img[:,:,i] = qdhe(img[:,:,i])
    he_img[:,:,i] = cv.equalizeHist(img[:,:,i])
    clahe_img[:,:,i] = clahe.apply(img[:,:,i])
    
    
  ## Show Images
  plt.figure(figsize=(8, 6))
  displayImage(img, qdhe_img, he_img, clahe_img)
  plt.show()
  
  #################### ASSESSMENT PART #####################
  # # AMBE Assessment
  # qdhe_ambe = assess.ambe(img, qdhe_img)
  # he_ambe = assess.ambe(img, he_img)
  # clahe_ambe = assess.ambe(img, clahe_img)
  # print(qdhe_ambe, he_ambe, clahe_ambe)
    
  # # Entropy Assessment
  # original_entropy = assess.entropy(img)
  # qdhe_entropy = assess.entropy(qdhe_img)
  # he_entropy = assess.entropy(he_img)
  # clahe_entropy = assess.entropy(clahe_img)
  # print(original_entropy, qdhe_entropy, he_entropy, clahe_entropy)
  
  # # Contrast Assessment
  # original_contrast = assess.contrast(img)
  # qdhe_contrast = assess.contrast(qdhe_img)
  # he_contrast = assess.contrast(he_img)
  # clahe_constrast = assess.contrast(clahe_img)
  # print(original_contrast, qdhe_contrast, clahe_img)
    

  return qdhe_img


## plot [0 255] histogram from image
def pltHist(im):
  plt.bar(np.arange(256), cv.calcHist([im.astype(np.uint8)], [0], None, [256], [0, 256]).flatten())

  
def displayImage(input, output, he_img=None, clahe_img=None, showHist=False):
  if sys.argv[2] == 'rgb':
    print("cmap = None")
    cmap = None
  else:
    print("cmap = gray")
    cmap ='gray'
  
  r = 1
  if showHist:
    r = 2
    
  plt.subplot(r, 4, 1)
  plt.imshow(input, cmap=cmap)
  plt.title('Original Image')
  plt.axis('off')
  if showHist:
    plt.subplot(2, 4, 5)
    pltHist(input)
    plt.title('Input Image Histogram')

  plt.subplot(r, 4, 2)
  plt.imshow(output, cmap=cmap, vmin=0, vmax=255)
  plt.title('QDHE Image')
  plt.axis('off')
  if showHist:
    plt.subplot(2, 4, 6)
    pltHist(output)
    plt.title('QDHE Histogram')
  
  if he_img.any() != None:
    plt.subplot(r, 4, 3)
    plt.imshow(he_img, vmin=0, vmax=255)
    plt.title('HE Image')
    plt.axis('off')
    if showHist:
      plt.subplot(2, 4, 7)
      pltHist(he_img)
      plt.title('HE Histogram')
  
  if clahe_img.any() != None:
    plt.subplot(r, 4, 4)
    plt.imshow(clahe_img, vmin=0, vmax=255)
    plt.title('CLAHE Image')
    plt.axis('off')
    if showHist:
      plt.subplot(2, 4, 8)
      pltHist(clahe_img)
      plt.title('CLAHE Histogram')

  plt.tight_layout()


## Input one channel of image, either gray, red, green, or blue
def qdhe(img, debug_display=False, debug = False):
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
  
  ## DEBUG
  if debug:
    print("Thresholds:", m)  
    print("span:", span)
    print("range:", range_)
    print("New Thresholds:", new_m)
  
  ## Allocate clipped data
  for i in range(4):
    clipped_cdf = np.cumsum(clipped_hist[m[i]:m[i+1]])
    try:
      normalized_cdf[m[i]:m[i+1]] = clipped_cdf / clipped_cdf[-1]
    except IndexError:
      normalized_cdf[m[i]:m[i+1]] = 0
    map_[m[i]:m[i+1]] = range_[i] * normalized_cdf[m[i]:m[i+1]] + new_m[i]

  # Fix overexposure spots
  max_idx = np.argmax(map_)
  map_[max_idx:] = 255

  # Histogram equalization
  QDHE_img = map_[img]
 
  
  ## Plot image
  if debug_display:
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
    
    # plt.step(np.arange(256), map_)
    # plt.show()
        
  return QDHE_img

if __name__ == "__main__":
  main()  