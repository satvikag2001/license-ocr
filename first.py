#%%
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

#%%
def data_prep(Path):
    global plate_img, plate_min_y, plate_min_x, plate_max_y, plate_max_x
    img_ori = cv2.imread(Path)
    
    height, width, channel = img_ori.shape
    
    #cv2.imshow("car", img_ori)
    
    #%%
    
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    
    #%%
    alpha = 1.5
    beta = 0
    
    gray = cv2.convertScaleAbs(gray, alpha = alpha, beta = beta)
    
    #%%
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
    
    #%%
    contours, _= cv2.findContours(
        img_thresh, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    
    #plt.imshow(temp_result)
    
    #%%
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    contours_dict = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    
    #plt.imshow(temp_result)
    #%%
    
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0
    
    possible_contours = []
    
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
            
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for d in possible_contours:
    #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        
    #%%
    
    MAX_DIAG_MULTIPLYER = 5 # 5
    MAX_ANGLE_DIFF = 12.0 # 12.0
    MAX_AREA_DIFF = 0.5 # 0.5
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 3 # 3
    
    def find_chars(contour_list):
        matched_result_idx = []
        
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue
    
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])
    
                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
    
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']
    
                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])
    
            # append this contour
            matched_contours_idx.append(d1['idx'])
    
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue
    
            matched_result_idx.append(matched_contours_idx)
    
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])
    
            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            
            # recursive
            recursive_contour_list = find_chars(unmatched_contour)
            
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
    
            break
    
        return matched_result_idx
        
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for r in matched_result:
        for d in r:
            #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)    
      
    #plt.imshow(temp_result)    
    #%%
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    
    for r in matched_result:
        for d in r:
            #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(img_ori, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0, 0, 255), thickness=2)
    
    #plt.imshow(img_ori)
    #%%
    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10
    
    plate_imgs = []
    plate_infos = []
    
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
    
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']
    
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
    
    #%%
    longest_idx, longest_text = -1, 0
    plate_chars = []
    
    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0
    
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            area = w * h
            ratio = w / h
    
            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h                    
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        
    def reverse(img):
        img = 255-img
        return img   
    img = reverse(img_result)
    
    #%%
    
    def find_contours(dimensions, img) :
    
        # Find all contours in the image
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        # Retrieve potential dimensions
        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]
        
        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
        
        ii = cv2.imread('contour.jpg')
        
        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs :
            # detects contour in binary image and returns the coordinates of rectangle enclosing it
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
            
            # checking the dimensions of the contour to filter out the characters by contour's size
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours
    
                char_copy = np.zeros((44,24))
                # extracting each character using the enclosing rectangle's coordinates.
                char = img[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))
                
                cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
                #plt.imshow(ii, cmap='gray')
                #plt.axis('off')
    
                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)
    
                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0
    
                img_res.append(char_copy) # List that stores the character's binary image (unsorted)
                
        # Return characters on ascending order with respect to the x-coordinate (most-left character first)
                
        #plt.show()
        # arbitrary function that stores sorted list of character indeces
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])# stores character images according to their index
        img_res = np.array(img_res_copy)
    
        return img_res
    
    #%%
    def segment_characters(image) :
    
        # Preprocess cropped license plate image
        img_lp = cv2.resize(image, (333, 75))
    
        LP_WIDTH = img_lp.shape[0]
        LP_HEIGHT = img_lp.shape[1]
    
        # Make borders white
        img_lp[0:3,:] = 255
        img_lp[:,0:3] = 255
        img_lp[72:75,:] = 255
        img_lp[:,330:333] = 255
    
        # Estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH/6,
                           LP_WIDTH/2,
                           LP_HEIGHT/10,
                           2*LP_HEIGHT/3]
            
        
    
        # Get contours within cropped license plate
        char_list = find_contours(dimensions, img_lp)
    
        return char_list

#%%
    char = segment_characters(img)
    return char

#%%
'''
import os
ori_path = "D:\\New folder"
path = 'archive/images/data'
os.chdir(path)
imagez = os.listdir()
images = []
for image in imagez:
    images.append(path+'\\'+image)
os.chdir(ori_path)

count = 0
cnt = 0
for i in images:
    cnt+=1
    print(cnt)
    chars = data_prep(i)
    for j in chars:
        count +=1
        name = "data/image"+str(count)+".png"
        cv2.imwrite(name,j)


'''

chars = data_prep("img.png")
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
    img = 255-img
    #cv2.imshow("image", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    img = img.reshape(1,32,32,1)
    
    prediction = model.predict(img)
    a = ["0","1","2","3","4","5","6","7",'8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y ','Z']
    b = np.argwhere(prediction ==1)
    
    
    return a[b[0,1]]

for char in chars:
    #cv2.imshow("plate",char)
    
    a = predict_char(char)
    #print(a) 
    final+=a

print(final)








