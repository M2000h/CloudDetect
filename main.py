import cv2
import tifffile as tiff
import numpy as np
filename = 'example.tiff'
a = tiff.imread(filename)
c = a.copy()
# b = np.ndarray(shape=(len(a), len(a[0]),3), dtype=int)
# a = a[((x[0] for y in a for x in y)=0)]
a = cv2.blur(a, (50,50))
b = np.all(a < (200, 200, 200), axis=-1)
c[b==0]=0

tiff.imsave('new.tiff', a)
tiff.imsave('new1.tiff', c)
tiff.imsave("output"+filename, c)
