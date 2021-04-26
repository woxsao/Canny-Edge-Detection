import numpy as np
import cv2 as cv
import math


class CannyEdge:
    def __init__(self, img, sigma, kernel_size, lowthresh_ratio, highthresh_ratio):
        self.img = img
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowthresh_ratio = lowthresh_ratio
        self.highthresh_ratio = highthresh_ratio

    def GaussianBlur(self):
        arr = np.ndarray((1,self.kernel_size))
        arr.fill(numintegrate(0.5,1.5, self.sigma))
        arrT = arr.T
        kernel = np.outer(arr,arrT)
        normalized_kernel = normalize(kernel)
        return normalized_kernel
    
    def gradient_calculation(self, smooth_img):
        kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
        ky = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], np.float32)
        convolved_x = cv.filter2D(smooth_img, ddepth = -1, kernel = kx, borderType= cv.BORDER_DEFAULT)
        convolved_y = cv.filter2D(smooth_img, ddepth = -1, kernel = ky, borderType= cv.BORDER_DEFAULT)
        gradient = np.hypot(convolved_x, convolved_y)
        normalized_g = gradient/gradient.max() * 255
        theta = np.arctan2(convolved_y,convolved_x)
        return normalized_g, theta 

    def non_max_suppression(self, gradient, theta_rad):
        #radian to angle conversion
        angle = theta_rad * 180. / math.pi
        #limit everything between 0-> 180 for ease
        angle[angle < 0] += 180
        x, y = gradient.shape
        new_arr = np.zeros((x,y), dtype=np.int32)
        
        for i in range(1,x-1):
            for j in range(1,y-1):
                a = 255
                b = 255        
                #angle 0 and angle 180 will both examine the pixels on either side, so we can put them in the same if statement. 
                if (0 <= angle[i,j] < 22.5) or (157 <= angle[i,j] <180):
                    #right
                    a = gradient[i, j+1]
                    #left
                    b = gradient[i, j-1]
                #angle 45 examines the pixel down and to the left, and the one up and to the right
                elif (22.5 <= angle[i,j] < 67.5) :
                    # down and to the left
                    a = gradient[i+1, j-1]
                    #up and to the right
                    b = gradient[i-1, j+1]
                #angle 90 examines the angle below and above
                elif (67.5 <= angle[i,j] < 112.5):
                    #below
                    a = gradient[i+1, j]
                    #above
                    b = gradient[i-1, j]
                #angle 135 examines the one up and to the left, and down and to the right
                elif (112.5 <= angle[i,j] < 157.5):
                    #up and to the left
                    a = gradient[i-1, j-1]
                    #down and to the right
                    b = gradient[i+1, j+1]

                if (gradient[i,j] >= a) and (gradient[i,j] >= b):
                    new_arr[i,j] = gradient[i,j]
                else:
                    new_arr[i,j] = 0
      
        return new_arr

    def double_thresholding(self, nms):
        high = self.highthresh_ratio * nms.max()
        low = high * self.lowthresh_ratio
        strong_pixel = 255
        weak_pixel = 40
        
        new_arr = np.zeros(nms.shape)

        strong_i, strong_j = np.where(nms >= high)
        weak_i, weak_j = np.where((nms < high) & (nms >= low))

        new_arr[strong_i, strong_j] = strong_pixel
        print(len(new_arr[strong_i, strong_j]))
        new_arr[weak_i,weak_j] = weak_pixel

        return new_arr

    def hysteresis(self, dt, weak = 40, strong = 255):
        x,y = dt.shape
        #new_arr = np.zeros((x,y))
        for i in range(1, x-1):
            for j in range(1, y-1):
                if dt[i,j] == weak:    
                    if((dt[i-1,j-1] == strong) or (dt[i-1,j] == strong) or (dt[i-1,j+1] == strong) 
                    or (dt[i,j-1] == strong) or (dt[i,j+1] == strong) or 
                    (dt[i+1,j-1] == strong) or (dt[i+1,j] == strong) or (dt[i+1,j+1] == strong)):
                        dt[i,j] = strong
                    else:
                        dt[i,j] = 0
        print(len(np.where(dt != 0)[0]))
        return dt

    def runner(self):
        convolved_img = cv.filter2D(src = self.img, ddepth = -1, kernel = self.GaussianBlur(), borderType=cv.BORDER_DEFAULT)
        gradient, theta = self.gradient_calculation(convolved_img)
        nms = self.non_max_suppression(gradient, theta)
        dt = self.double_thresholding(nms)
        dt_dup = np.copy(dt)
        hyster = self.hysteresis(dt =dt)
        return dt_dup, hyster
        


def gaussx(x, mu, sigma):
        coefficient = 1/(sigma*(math.pi*2)**0.5)
        exp = -0.5*(x-mu)**2/sigma**2
        gauss = coefficient*math.exp(exp)
        return gauss
def numintegrate(a,b, sigma):
        pointer = a
        interval = 0.000001
        sum = 0
        while pointer <= b:
            gauss = gaussx(pointer ,0.0, sigma)
            sum += gauss 
            pointer += interval 
        return sum
def normalize(arr):
        sum = arr.sum()
        new_arr = np.true_divide(arr, sum)
        return new_arr

