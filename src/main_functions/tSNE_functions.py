import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import PIL.Image


def print_letter_variations(data):
  #digits = datasets.load_digits(n_class=6)
  X = np.array(data[:100].copy())

  # Plot images of the digits
  n_img_per_row = 10
  img = np.zeros((128 * n_img_per_row, 128 * n_img_per_row))
  for i in range(n_img_per_row):
      ix = 128 * i
      for j in range(n_img_per_row):
          iy = 128 * j
          img[ix:ix + 128, iy:iy + 128] = X[i * n_img_per_row + j]
    
  img = PIL.Image.fromarray(np.uint8(img),)
  return img


def tsne_for_two(encodings):
  tsne = TSNE(n_components=2,
            init='pca',random_state=0, perplexity=30.0)
  return tsne.fit_transform(encodings)



def print_letter_cloud(data, tSNE_encodings):

  images = data*255

  tSNE_encodings -= np.min(tSNE_encodings, axis=0)
  M = 10000
  tSNE_encodings *= M / np.max(tSNE_encodings)
  canvas = PIL.Image.new('L', (M + 256, M + 128), 255)
  for i in range(len(tSNE_encodings)):
      cx, cy = map(int, tSNE_encodings[i])
      img_black = PIL.Image.new('L', (128, 128), 0)
      img = PIL.Image.fromarray(np.invert(np.uint8(images[i])),)
      canvas.paste(img_black, (cx, cy), img)
  
  return canvas






 
