import tifffile as tiff
a = tiff.imread('example.tiff')
a[a > 150] = 0
tiff.imsave('new.tiff', a)