from . import CannyEdge
#from CannyEdge import CannyEdge
import cv2 as cv
from matplotlib import pyplot as plt


#print(sys.path)
opera_house = cv.imread('/Users/MonicaChan/Desktop/AT/CV unit/cannyedgedetectionproject/SydneyOperaHouse.jpg', cv.IMREAD_GRAYSCALE)
plt.imshow(opera_house)
plt.show()


x = CannyEdge.CannyEdge(opera_house,sigma = 1, kernel_size = 5, lowthresh_ratio = 0.02, highthresh_ratio = 0.08)
dt, hyster = x.runner()
plt.imshow(dt)
plt.show()
plt.imshow(hyster)
plt.show()

opera_house = cv.imread('/Users/MonicaChan/Desktop/AT/CV unit/cannyedgedetectionproject/lena512.png', cv.IMREAD_GRAYSCALE)
plt.imshow(opera_house)
plt.show()


x = CannyEdge.CannyEdge(opera_house,sigma = 1, kernel_size = 5, lowthresh_ratio = 0.08, highthresh_ratio = 0.11)
dt, hyster = x.runner()
plt.imshow(dt)
plt.show()
plt.imshow(hyster)
plt.show()