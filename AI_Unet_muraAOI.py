from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tqdm.auto import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import time


IMAGE_LIB = 'input_img/*.png'
img_paths = sorted(glob(IMAGE_LIB))

IMG_SIZEx = 456
IMG_SIZEy = 256
NUM_CLASSES = 4

#======建立x_data(通道1)及x_input_data(通道3)
x_data = np.empty((len(img_paths), IMG_SIZEy, IMG_SIZEx))
print(x_data.shape)
x_input_data = np.empty((len(img_paths), IMG_SIZEy, IMG_SIZEx, 3))
print(x_input_data.shape)


#======將資料夾路徑裡的png圖片分別做一份通道1的datasets(x_data),及通道3的datasets(x_input_data)
for i, path in enumerate(tqdm(img_paths)):
    #print(path)
    # read input image
    input_img = cv2.imread(path)
    input_img = cv2.resize(input_img, (IMG_SIZEx, IMG_SIZEy))

    img = cv2.imread(path)[:,:,0] # get channel 0 since it's a gray scale image
    img = cv2.resize(img, (IMG_SIZEx, IMG_SIZEy))
    img = img / 255

    x_data[i] = img
    x_input_data[i] = input_img


#=====針對通道1的datasets 用訓練好的Unet模型預測找出Defect並利用openCV找出特徵點
n = 10
dilate_n = 5

x_data = np.expand_dims(x_data, axis=-1)
print(x_data.shape)
img_input = x_data[n:n+1]
cv2.imshow("Input_img", x_data[n])

model = load_model("multi-class-seg_4.h5")
y_pred = model.predict(img_input)

a_list = []
for k in range(1, NUM_CLASSES):
    mask2 = y_pred[0, :, :, k]
    mask2[mask2>=0.5] = 255
    mask2[mask2<=0.5] = 0
    m2 = mask2
    # print(m2)
    m2 = np.array(m2,np.uint8)
#==============
    for _ in range(dilate_n):
        m2 = cv2.dilate(m2, (IMG_SIZEy, IMG_SIZEx))

#================
    a, b=cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for num in a:
        a_list.append(num)
print(len(a_list))


#=====將特徵點用紅框畫在3通道圖片上，框出Defect位置
m1 = x_input_data[n]
m1 = np.array(m1,np.uint8)

for j in range(len(a_list)):
    x, y, w, h = cv2.boundingRect(a_list[j])
    cv2.rectangle(m1, (x, y), (x+w, y+h), (0, 0, 255), 2)
cv2.imshow("result 1", m1)

cv2.waitKey(0)
cv2.destroyAllWindows()

#=====將模型預測出來的圖形畫出來
fig, ax = plt.subplots(1,4,figsize=(12,4))
for i in range(NUM_CLASSES):
    for _ in range(dilate_n):
        y_truth = cv2.dilate(y_pred[0, :, :, i], (IMG_SIZEy, IMG_SIZEx))
        y_pred[0, :, :, i] = y_truth
    ax[i].imshow(y_truth, cmap='gray')  
plt.show()



    






