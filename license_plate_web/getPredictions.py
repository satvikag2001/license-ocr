import cv2
import numpy as np
def predict_char(char):
    img = char
    #img = cv2.imread("data2/data2/training_data/1/44405.png")
     
    #print(img.shape)
    #cv2.imshow("image", img)
    from tensorflow import keras
    model = keras.models.load_model("try3.h5")
    if img.shape[-1]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(32,32),interpolation = cv2.INTER_CUBIC)
    #cv2.ims               how("image", img)
    #_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #_,img = cv2.threshold(img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #cv2.rectangle(img,(0,0),(32,32),(255,255,255),9)
    #cv2.imshow("img", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    img = img.reshape(1,32,32,1)
    #img = img//255
    prediction = model.predict(img)
    letters = ["0","1","2","3","4","5","6","7",'8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y ','Z']
    b = np.argwhere(prediction ==1)
    #print(letters[b[0,1]])
    return letters[b[0,1]]
    
def getPredictions(path):

#%%
    img = cv2.imread(path)
    from tensorflow import keras
    model1 = keras.models.load_model("try2.h5")
    img = cv2.resize(img, (200,200))
    X = np.array(img)
    X = X.reshape((1,200,200,3))
    X_2 = X/255
    dimen = model1.predict(X_2)
    dimen = dimen*255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    (thresh, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    #cv2.imshow("asdf",bw)
    a1 = int(dimen[0,0])
    a2 = int(dimen[0,1])
    a3 = int(dimen[0,2])
    a4 = int(dimen[0,3])
    #bw = cv2.rectangle(bw,(a1,a2),(a3,a4), color = (127,127,127))
    #cv2.imshow("asdf",bw)
    crop =  bw[a4:a2,a3:a1]
    crop2 = img[a4:a2,a3:a1]
    #crop = 255-crop
    crop = cv2.resize(crop, (333,75), interpolation = cv2.INTER_CUBIC)
    crop2 = cv2.resize(crop2,(333,75), interpolation = cv2.INTER_CUBIC)
    #crop = cv2.GaussianBlur(crop, ksize=(3, 3), sigmaX=0)
    #_, crop = cv2.threshold(crop, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #crop = cv2.copyMakeBorder(crop, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    chars = []
    numb= 7
    for i in range(numb):
        chars.append(crop[:,int(333/numb)*i:int(333/numb)*(i+1)])
    #cv2.imshow("asdf",crop)
    final = ""
    for i in range(len(chars)):
        abc = predict_char(chars[i])
        final+=abc
    return final
