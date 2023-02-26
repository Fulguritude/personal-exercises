import sys
sys.path.insert(1, '../ex01/')

from ScrapBooker import ScrapBooker as sb
import ImageProcessor as ip

imgprc = ip.ImageProcessor()
img = imgprc.load("../42ai.png")


#img = sb.crop(img, (80,100), (70, 50))
#img = sb.thin(img, 3, 1)
#img = sb.juxtapose(img, 3, 0)
img = sb.mosaic(img, (3,2))

imgprc.display(img)