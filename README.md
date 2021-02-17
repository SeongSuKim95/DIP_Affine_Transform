# DIP_AffineTransform & Color Transform
## This repository is python code of "Digital Image Processing, Rafael C.Gonzalez, 3rd edition"

* Intensity Transformation
  * Negative transformation
    ![Negative_transform](https://user-images.githubusercontent.com/62092317/108153340-50979100-711e-11eb-9fa6-cce9bf82dcfe.png)
     ```python
     def NegativeTransformation(img):
         negative_transform = np.array(255-img,dtype='uint8')
         return negative_transform
     ```
  * Gamma transformation
     ```python
      def GammaTransformation(img,gamma):
          c=1
          Gamma_transform = c*np.array(255*(img/255)**gamma,dtype='uint8')
          return Gamma_transform
     ```
     
  * Log transformation
    ![Log_transform](https://user-images.githubusercontent.com/62092317/108153338-4f666400-711e-11eb-933c-b42ee5301e54.PNG)
      ```python
      def logtransformation(img):
          c=255/np.log(256)/np.log(8)
          log_transform = c*np.log(1+img)/np.log(8)
          result = np.array(log_transform, dtype=np.uint8)
          return result
      ```
  
  * ContrastStretching
    ![Contrast_Stretching](https://user-images.githubusercontent.com/62092317/108153357-58efcc00-711e-11eb-87ba-51a03ac23905.png)
      ```python
      def ContrastStretching(img):
          original = np.array(img)
          min = np.min(original)
          max = np.max(original)
          PiecewiseLinear = np.zeros(256, dtype=np.uint8)
          PiecewiseLinear[min:max+1] = np.linspace(0, 255, max-min+1,True,dtype=np.uint8)
          result = np.array(PiecewiseLinear[original],dtype=np.uint8)
          return result
      ```
  * Bitslicing
    ![Bit_Slicing](https://user-images.githubusercontent.com/62092317/108153354-57be9f00-711e-11eb-8140-8e45e45aa0f9.png)
      ```python
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

          return final
      ``

* Affine Transform
  ![Scaling_Interpolation](https://user-images.githubusercontent.com/62092317/108153352-57260880-711e-11eb-9cd3-52dfdef08761.png)
  * Bilinear interpolation
  ```python
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

       if a:
           result[2] = Original[y1,x1]
           transformed[y,x] = result[2]

       else :
           result[2] = (Original[y1,x2]-Original[y1,x1])*(result[0]-x1)+Original[y1,x1]
           transformed[y,x] = int(result[2])

      result = np.array(transformed, dtype=np.uint8)
      return result

  def backward_bilinear_y(img,c):
      Original = np.array(img,dtype='int64')
      transformed = np.zeros(shape=(int(c*(Original.shape[0])),int((Original.shape[1]))),dtype='int64')

      cy = (c*Original.shape[0]-1)/(Original.shape[0]-1)
      zoom_reversematrix = np.array([[1,0,0],[0,1/cy,0],[0,0,1]])
      for x,y,element in enumerate2(transformed):
       tuple= (x,y,element)
       array=np.asarray(tuple)
       result = array.dot(zoom_reversematrix)
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
  ```    
  * Nearest interpolation
  ```python
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

       if a:
           result[2] = Original[y1,x1]
           transformed[y,x] = result[2]

       else :
          if result[0]-x1 > 0.5:
           result[2] = Original[y1,x2]
           transformed[y,x] = int(result[2])
          else :
           result[2] = Original[y1,x1]
           transformed[y,x] = int(result[2])

      result = np.array(transformed, dtype=np.uint8)
      return result

  def backward_nearest_y(img,c):
      Original = np.array(img,dtype='int64')

      transformed = np.zeros(shape=(int(c*(Original.shape[0])),int((Original.shape[1]))),dtype='int64')

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
  ``` 
  
  * Rotation
  ![Rotation_transform](https://user-images.githubusercontent.com/62092317/108153350-55f4db80-711e-11eb-8446-418044dc1433.png)
  ```python
  def rotate_image(img,theta):

    img_height, img_width = img.shape
    corner_x, corner_y = rotatecoordination([0,img_width,img_width,0],[0,0,img_height,img_height],theta)
    destination_width, destination_height = (int(np.ceil(c.max()-c.min())) for c in (corner_x, corner_y))
    destination_x, destination_y = np.meshgrid(np.arange(destination_width), np.arange(destination_height))
    sx, sy = rotatecoordination(destination_x + corner_x.min(), destination_y + corner_y.min(), -theta)
    sx, sy = sx.round().astype(int), sy.round().astype(int)
    valid = (0 <= sx) & (sx < img_width) & (0 <= sy) & (sy < img_height)
    transformed=np.zeros(shape=(destination_height, destination_width), dtype=img.dtype)
    transformed[destination_y[valid], destination_x[valid]] = img[sy[valid], sx[valid]]
    transformed[destination_y[~valid], destination_x[~valid]]=0

    return transformed
  ```
  
* Color Transform
  ![RGB_to_HSI](https://user-images.githubusercontent.com/62092317/108153344-52615480-711e-11eb-92ce-e75d56e27ddd.png)
  * RGB_to_HIS
   ```python 
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
   ```
  * RGB_to_Ycbcr
   ![RGB_to_Ycbcr](https://user-images.githubusercontent.com/62092317/108153346-542b1800-711e-11eb-8776-01c3af86fd33.png)
   ```python
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
   ```
   
   * Ycbcr_to_RGB
   
   ```python
   def Ycrbr_to_RGB(image):
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
   ```
   
   * RGB_to_CMY
   ![RGB_to_CMY](https://user-images.githubusercontent.com/62092317/108153343-51302780-711e-11eb-8dc4-ea0e324a3f89.png)
   ```python
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
   ```
