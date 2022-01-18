import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('base_de_données.csv')
csv_thermiques = df['nom_fichier']
sift = cv2.SIFT_create()

data = pd.read_csv('./thermiques/' + csv_thermiques[97] ,delimiter=';')
img = data.to_numpy()
#shape (287,383)
img = img[:,:-1]
#shape (287,382)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		L = img[i,j].split(',')
		if len(L) == 2:
			img[i,j] = L[0] + '.' + L[1]
		else:
			img[i,j] = L[0] + '.' + '0'

img = img.astype(np.float64)
mask = (img > 60)

min = np.min(img[mask])
max = np.max(img[mask])



print(min)
print(max)

img = img - min
img = ( img / (max-min) ) * 255

image = mask*img
image = np.round(image,decimals=0)
image = image.astype(np.uint8)
print(image.dtype)
mask = np.array(np.uint8(mask))

cv2.imwrite('test.png',image)
img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)

#image est désormais prête à être utilisée par opencv

# cv2.imshow('img',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

kp = sift.detect(img,mask=mask)
img_=cv2.drawKeypoints(img,kp,img)
cv2.imwrite('sift_thermique.jpg',img_)

