import sys
sys.path.insert(1, '../ex01/')

from ColorFilter import ColorFilter as cf
import ImageProcessor as ip

imgprc = ip.ImageProcessor()
img = imgprc.load("../42ai.png")


#img = cf.celluloid(img)
img = cf.to_grayscale(img, 'm')

imgprc.display(img)