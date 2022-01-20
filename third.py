import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
def symbol_id_to_symbol(symbol_id = None):
    if symbol_id:
        symbol_data = SYMBOLS.loc[SYMBOLS['symbol_id'] == symbol_id]
        if not symbol_data.empty:
            return str(symbol_data["latex"].values[0])
        else:
            print("This should not have happend, wrong symbol_id = ", symbol_id)
            return None
    else: 
        print("This should not have happend, no symbol id passed")
        return None
from first import *
SYMBOLS = pd.read_csv("datta/symbols.csv") 
SYMBOLS = SYMBOLS[["symbol_id", "latex"]]
#chars = data_prep("image.png")
final = ""
def predict_char(char):
    img = char
    #print(img.shape)
    #cv2.imshow("image", img)
    from tensorflow import keras
    model = keras.models.load_model("try1.h5")
    if img.shape[-1]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.resize(img,(32,32),interpolation = cv2.INTER_CUBIC)
    #cv2.ims               how("image", img)
    
    img = img.reshape(1,32,32,1)
    
    prediction = model.predict(img)
    a = ["0","1","2","3","4","5","6","7",'8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y ','Z']
    b = np.argwhere(prediction ==1)
    print(a[b[0,1]])
    
    

char = cv2.imread("data/image53.png")
predict_char(char)
 
