import numpy as np
import cv2
import math

def main():

    img = cv2.imread('image_lenna.jpg',cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('original_image',img)
    #cv2.imshow('Original_image',img)
    #cv2.waitKey(0)
    #imgHIS=cv2.imread('imageHIS.jpg',cv2.IMREAD_COLOR)
    #HIS_RGBtransformed = HIS_RGBtransformation(imgHIS)
    #cv2.imwrite('HIS2RGB.jpg',HIS_RGBtransformed)
    #cv2.imshow('HIS_RGBtransformed',imgHIS)
    cv2.imshow('Original',img)
    cv2.waitKey(0)
    #RGB_HIScrtransformed = RGB_to_HIS(img)
    #cv2.imshow('RGB_to_HIS',RGB_HIScrtransformed)
    bitsliced = bitslicing(img)
    cv2.imshow('87slice',bitsliced)
    cv2.imwrite('bit slcing.jpg',bitsliced)
    #imgRGB = cv2.imread('ImageRGB.jpg',cv2.IMREAD_COLOR)
    #RGB_HIStransformed=RGB_HIStransformation(imgRGB)
    #cv2.imshow('RGB_HIStransformed',RGB_HIStransformed)
    #cv2.imwrite('RGB2HIS.jpg',RGB_HIStransformed)

    #negative_transformed = NegativeTransformation(img)
    #cv2.imshow('negative_transformed',negative_transformed)
    #cv2.waitKey(0)

    #contrast_transformed = ContrastStretching(img)
    #cv2.imshow('contrast_transformed',contrast_transformed)

    #Log_transformed = logtransformation(img)
    #cv2.imshow('Log_transformed',Log_transformed)

    #zoom_transformed=backward_bilinear_x(backward_bilinear_y(img,4),4)
    #cv2.imwrite('bilinear interpolation image.jpg',zoom_transformed)

    #rotate_transform = rotate_image(img,110*np.pi/180)
    #cv2.imshow('rrr',rotate_transform)

    cv2.waitKey(0)

def NegativeTransformation(img):
    negative_transform = np.array(255-img,dtype='uint8')
    return negative_transform
def GammaTransformation(img,gamma):
    c=1
    Gamma_transform = c*np.array(255*(img/255)**gamma,dtype='uint8')
    return Gamma_transform

def logtransformation(img):
    c=255/np.log(256)/np.log(8)
    #c=(1/2)*255/np.log(1+np.max(img))/np.log(np.sqrt(10))
    log_transform = c*np.log(1+img)/np.log(8)
    result = np.array(log_transform, dtype=np.uint8)
    return result

def ContrastStretching(img):
    original = np.array(img)
    min = np.min(original)
    max = np.max(original)
    PiecewiseLinear = np.zeros(256, dtype=np.uint8)
    PiecewiseLinear[min:max+1] = np.linspace(0, 255, max-min+1,True,dtype=np.uint8)
    result = np.array(PiecewiseLinear[original],dtype=np.uint8)
    return result

def bitslicing(img):

    lst = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst.append(np.binary_repr(img[i][j], width=8))

    eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
    seven_bit_img = (np.array([int(i[1]) for i in lst], dtype=np.uint8) * 64).reshape(img.shape[0], img.shape[1])
    six_bit_img = (np.array([int(i[2]) for i in lst], dtype=np.uint8) * 32).reshape(img.shape[0], img.shape[1])
    five_bit_img = (np.array([int(i[3]) for i in lst], dtype=np.uint8) * 16).reshape(img.shape[0], img.shape[1])
    four_bit_img = (np.array([int(i[4]) for i in lst], dtype=np.uint8) * 8).reshape(img.shape[0], img.shape[1])
    three_bit_img = (np.array([int(i[5]) for i in lst], dtype=np.uint8) * 4).reshape(img.shape[0], img.shape[1])
    two_bit_img = (np.array([int(i[6]) for i in lst], dtype=np.uint8) * 2).reshape(img.shape[0], img.shape[1])
    one_bit_img = (np.array([int(i[7]) for i in lst], dtype=np.uint8) * 1).reshape(img.shape[0], img.shape[1])

    finalr = cv2.hconcat([eight_bit_img, seven_bit_img, six_bit_img, five_bit_img])
    finalv = cv2.hconcat([four_bit_img, three_bit_img, two_bit_img, one_bit_img])
    final78 = eight_bit_img + seven_bit_img
    final = cv2.hconcat([finalr, finalv])

    return final78
#affine Transform
def backward_bilinear_x(img,c):
    Original = np.array(img,dtype='int64')
    transformed = np.zeros(shape=(int((Original.shape[0])),int(c*(Original.shape[1]))),dtype='int64')
    cx = (c*Original.shape[1]-1)/(Original.shape[1]-1)
    zoom_reversematrix = np.array([[1/cx,0,0],[0,1,0],[0,0,1]])

    for x,y,element in enumerate2(transformed):
     tuple= (x,y,element)
     array=np.asarray(tuple)
     result=array.dot(zoom_reversematrix)
     x1=int(result[0])
     y1=int(result[1])
     x2=x1+1
     a=result[0].is_integer()
     #print(x,y)
     if a:
         result[2] = Original[y1,x1]
         transformed[y,x] = result[2]
         #print(result)
         #print(transformed)
     else :
         result[2] = (Original[y1,x2]-Original[y1,x1])*(result[0]-x1)+Original[y1,x1]
         transformed[y,x] = int(result[2])
         #print(result)
         #print(transformed)
    result = np.array(transformed, dtype=np.uint8)
    return result

def backward_bilinear_y(img,c):
    Original = np.array(img,dtype='int64')
    #test_array = np.arange(8).reshape(2,4)
    #print(test_array)
    transformed = np.zeros(shape=(int(c*(Original.shape[0])),int((Original.shape[1]))),dtype='int64')
    #print(transformed)
    #cx = (c*test_array.shape[1]-1)/(test_array.shape[1]-1)
    cy = (c*Original.shape[0]-1)/(Original.shape[0]-1)
    zoom_reversematrix = np.array([[1,0,0],[0,1/cy,0],[0,0,1]])
    for x,y,element in enumerate2(transformed):
     tuple= (x,y,element)
     array=np.asarray(tuple)
     #print(array)
     result = array.dot(zoom_reversematrix)
     #print(result)
     x1=int(result[0])
     y1=int(result[1])
     y2=y1+1
     a=result[1].is_integer()

     if a:
         result[2] = Original[y1,x1]
         transformed[y,x] = result[2]

     else :
         result[2] = (Original[y2,x1]-Original[y1,x1])*(result[1]-y1)+Original[y1,x1]
         transformed[y,x] = result[2]

    result = np.array(transformed, dtype=np.uint8)
    return result

def backward_nearest_x(img,c):
    Original = np.array(img,dtype='int64')
    transformed = np.zeros(shape=(int((Original.shape[0])),int(c*(Original.shape[1]))),dtype='int64')
    cx = (c*Original.shape[1]-1)/(Original.shape[1]-1)
    zoom_reversematrix = np.array([[1/cx,0,0],[0,1,0],[0,0,1]])

    for x,y,element in enumerate2(transformed):
     tuple= (x,y,element)
     array=np.asarray(tuple)
     result=array.dot(zoom_reversematrix)
     x1=int(result[0])
     y1=int(result[1])
     x2=x1+1
     a=result[0].is_integer()
     #print(x,y)
     if a:
         result[2] = Original[y1,x1]
         transformed[y,x] = result[2]
         #print(result)
         #print(transformed)
     else :
        if result[0]-x1 > 0.5:
         result[2] = Original[y1,x2]
         transformed[y,x] = int(result[2])
        else :
         result[2] = Original[y1,x1]
         transformed[y,x] = int(result[2])
         #print(result)
         #print(transformed)
    result = np.array(transformed, dtype=np.uint8)
    return result


def backward_nearest_y(img,c):
    Original = np.array(img,dtype='int64')
    #test_array = np.arange(8).reshape(2,4)
    #print(test_array)
    transformed = np.zeros(shape=(int(c*(Original.shape[0])),int((Original.shape[1]))),dtype='int64')
    #print(transformed)
    #cx = (c*test_array.shape[1]-1)/(test_array.shape[1]-1)
    cy = (c*Original.shape[0]-1)/(Original.shape[0]-1)
    zoom_reversematrix = np.array([[1,0,0],[0,1/cy,0],[0,0,1]])
    for x,y,element in enumerate2(transformed):
     tuple= (x,y,element)
     array=np.asarray(tuple)
     #print(array)
     result = array.dot(zoom_reversematrix)
     #print(result)
     x1=int(result[0])
     y1=int(result[1])
     y2=y1+1
     a=result[1].is_integer()

     if a:
         result[2] = Original[y1,x1]
         transformed[y,x] = result[2]

     else :
         if result[1] - y1 > 0.5:
             result[2] = Original[y2, x1]
             transformed[y, x] = int(result[2])
         else:
             result[2] = Original[y1, x1]
             transformed[y, x] = int(result[2])
             transformed[y,x] = result[2]

    result = np.array(transformed, dtype=np.uint8)
    return result

def enumerate2(np_array):

    for y, row in enumerate(np_array):
        for x, element in enumerate(row):
            yield (x, y, element)

def rotatecoordination(x,y,theta):
    theta = -theta
    sin=np.sin(theta)
    cos=np.cos(theta)
    x=np.asarray(x)
    y=np.asarray(y)
    return x*cos-y*sin, x*sin+y*cos

def rotate_image(img,theta):
    # Dimensions of source image. Note that scipy.misc.imread loads
    # images in row-major order, so src.shape gives (height, width).
    img_height, img_width = img.shape
    # Rotated positions of the corners of the source image.
    corner_x, corner_y = rotatecoordination([0,img_width,img_width,0],[0,0,img_height,img_height],theta)

    destination_width, destination_height = (int(np.ceil(c.max()-c.min())) for c in (corner_x, corner_y))

    destination_x, destination_y = np.meshgrid(np.arange(destination_width), np.arange(destination_height))

    sx, sy = rotatecoordination(destination_x + corner_x.min(), destination_y + corner_y.min(), -theta)

    sx, sy = sx.round().astype(int), sy.round().astype(int)

    valid = (0 <= sx) & (sx < img_width) & (0 <= sy) & (sy < img_height)
    # Create destination image.
    transformed=np.zeros(shape=(destination_height, destination_width), dtype=img.dtype)
    # Copy valid coordinates from source image.
    transformed[destination_y[valid], destination_x[valid]] = img[sy[valid], sx[valid]]
    # Fill invalid coordinates.
    transformed[destination_y[~valid], destination_x[~valid]]=0

    return transformed

def RGB_to_HIS(img):
    with np.errstate(divide='ignore', invalid='ignore'):
        zmax = 255
        bgr = np.float32(img) / 255
        R= bgr[:, :, 2]
        G= bgr[:, :, 1]
        B= bgr[:, :, 0]

        a = (0.5) * np.add(np.subtract(R, G), np.subtract(R, B))
        b = np.sqrt(np.add(np.power(np.subtract(R, G), 2), np.multiply(np.subtract(R, B), np.subtract(G, B))))
        tetha = np.arccos(np.divide(a, b, out=np.zeros_like(a), where=b != 0))
        H = (180 / math.pi) * tetha
        H[B > G] = 360 - H[B > G]

        a = 3 * np.minimum(np.minimum(R, G), B)
        b = np.add(np.add(R, G), B)
        S = np.subtract(1, np.divide(a, b, out=np.ones_like(a), where=b != 0))

        I = (1 / 3) * np.add(np.add(R, G), B)
        stack = np.dstack((H, zmax * S, np.round(zmax * I)))
        result= np.array(stack,dtype=np.uint8)
        return result

def RGB_to_Ycbcr(image):
        img=(image.astype(float)/255)
        YCbCr_img = np.empty((img.shape[0], img.shape[1], 3), float)
        Y = np.empty([img.shape[0], img.shape[1]], dtype=float)
        Cb = np.empty([img.shape[0], img.shape[1]], dtype=float)
        Cr = np.empty([img.shape[0], img.shape[1]], dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                Y[i, j] = (0.299) * (img[i, j][2]) + (0.587) * (img[i, j][1]) + (0.114) * (img[i, j][0])
                Cb[i, j] = (-0.1687) * (img[i, j][2]) + (-0.3313) * (img[i, j][1]) + (0.5) * (img[i, j][0])
                Cr[i, j] = (0.5) * (img[i, j][2]) + (-0.4187) * (img[i, j][1]) + (-0.0813) * (img[i, j][0])
        YCbCr_img[..., 0]=Cr*255
        YCbCr_img[..., 1]=Cb*255
        YCbCr_img[..., 2]=Y*255
        result = np.array(YCbCr_img, dtype=np.uint8)

        return result

def ycrbr_to_RGB(image):
    img = (image.astype(float)/255)
    RGB_img = np.empty((img.shape[0], img.shape[1], 3), float)
    r = np.empty([img.shape[0],img.shape[1]], dtype = float)
    g = np.empty([img.shape[0],img.shape[1]], dtype = float)
    b = np.empty([img.shape[0],img.shape[1]], dtype = float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r[i,j] = (1)*(img[i,j][2]) + (0)*(img[i,j][1]) + (1.402)*(img[i,j][0])
            g[i,j] = (1)*(img[i,j][2]) + (-0.34414)*(img[i,j][1]) + (-0.71414)*(img[i,j][0])
            b[i,j] = (1)*(img[i,j][2]) + (1.772)*(img[i,j][1]) + (0)*(img[i,j][0])
    RGB_img[...,0] = b*255
    RGB_img[...,1] = g*255
    RGB_img[...,2] = r*255
    return RGB_img

def RGB_to_CMY(image):
        img = (image.astype(float) / 255)
        CMY_img = np.empty((img.shape[0], img.shape[1], 3), float)
        C = np.empty([img.shape[0], img.shape[1]], dtype=float)
        M = np.empty([img.shape[0], img.shape[1]], dtype=float)
        Y = np.empty([img.shape[0], img.shape[1]], dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                C[i, j] =1- (img[i, j][2])
                M[i, j] =1- (img[i, j][1])
                Y[i, j] =1- (img[i, j][0])
        CMY_img[..., 0] = C * 255
        CMY_img[..., 1] = M * 255
        CMY_img[..., 2] = Y * 255
        result = np.array(CMY_img, dtype=np.uint8)

        return result

main()