# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:52:05 2018

@author: Cristian
"""

import socket               # Import socket module
import struct
import cv2
import numpy as np

import pickle
import os
import shutil
import imutils
import operator

from sklearn.neural_network import MLPClassifier
from Alphabet_Recognizer_DL.Alphabet_Recognizer_DL import predict_for_cv


tosend = []
characters = []
charactersCopy = []
letter_count = {0: 'CHECK', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
                11: 'k',
                12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v',
                23: 'w',
                24: 'x', 25: 'y', 26: 'z', 27: 'CHECK'}


average_w = 0
def binarize(pathToFile, tipo):
    image = cv2.imread(pathToFile)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#            gray_image = cv2.resize(gray_image, None,2,2,cv2.INTER_CUBIC)
    gray2_image=gray_image.copy()
    
    if tipo==0:# daca fonul e deschis
        print('pppp')
        gray2_image = cv2.adaptiveThreshold(gray2_image,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 59,13) 
        white=np.count_nonzero(gray2_image.ravel())
        if white>len(gray2_image.ravel())-white:
            gray2_image = cv2.bitwise_not(gray2_image)
    elif tipo==1:#daca fonul e inchis
        gray2_image = cv2.adaptiveThreshold(gray2_image,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 51,3) 
        white=np.count_nonzero(gray2_image.ravel())
        if white<len(gray2_image.ravel())-white:
            gray2_image = cv2.bitwise_not(gray2_image)
    else:#daca fonul e o culoare(rosu/albastru)
        temp,gray2_image = cv2.threshold(gray2_image,0,255,
                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        white=np.count_nonzero(gray2_image.ravel())
        if white>len(gray2_image.ravel())-white:
            gray2_image = cv2.bitwise_not(gray2_image)

    gray2_image=cv2.medianBlur(gray2_image, 3)#????????????????????????????????????
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    gray2_image = cv2.dilate(gray2_image, kernel, iterations=1)    #kernel1 = np.ones((3,3),np.uint8)
        
    cv2.imwrite(pathToFile,gray2_image)
def segment_chars(pathToFile):
    global average_w
    global characters
    img = cv2.imread(pathToFile)
    blackim = img.copy()
    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    blackim = thresh.copy()  
    ret,blackim = cv2.threshold(thresh,0,0,cv2.THRESH_BINARY)
    
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
    
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []
    bigCnts=[]
    iteration=0
    
    w_minlimit = 15#*2
    w_maxlimit = 60#*2
    h_minlimit = 20#*2
    h_maxlimit = 90#*2
    average_h = 0
    num_elem = 0
    average_w = 0
    max_area = 0
    for c in cnts:
    	# compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= w_minlimit and w <= 2*w_maxlimit) and (h >= h_minlimit and h <= h_maxlimit):
            average_h+=h
            average_w+=w
            num_elem+=1
            bigCnts.append(c)
#        todo: create a new contours list without small elements
        if (w*h > max_area) and (w >= w_minlimit and w <= w_maxlimit) and (h >= h_minlimit and h <= h_maxlimit):
            max_area = w*h
    average_h=int(average_h/num_elem)
    average_w=int(average_w/num_elem)
    
    print(str(max_area))
    # loop over the digit area candidates
    for c in bigCnts:
    	# compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
     
    	# if the contour is sufficiently large, it must be a digit
        if (w >= average_w-int(50*average_w/100) and w<=3*average_w+5) and (h >= average_h and h <= average_h*2):# and(h*100>=2*w*65) and (w*h*2*100>=max_area*65)*/:
            digitCnts.append(c)
#        if (w >= w_minlimit and w<=w_maxlimit) and (h >= h_minlimit and h <= h_maxlimit):# and(h*100>=2*w*65) and (w*h*2*100>=max_area*65)*/:
#            digitCnts.append(c)
    iteration=0
    averageh = 0
    maxh = 0
    numbr = 0
    for c in digitCnts:
        numbr+=1
        (x, y, w, h) = cv2.boundingRect(c)
        averageh+=h
        if h>maxh:
            maxh=h
    ndigitCnts=[]
    averageh=int(averageh/numbr)
    
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if h>=averageh-5:
            ndigitCnts.append(c)
    for c in ndigitCnts:
    	# compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if 1.75*average_w-5<=w<=3*average_w:
            ret=0
            cropped = img[y :y + h , x : x + int(w/2)].copy()#img[y :y + h , x : x + w]
                #cv2.imshow('color',cropped)
    #        s = 'C:\\Users\\Cristian\\Desktop\\forlearn\\characters\\'+str(y)+'_'+str(x)+'.jpg' 
            iteration=iteration+1
            temp = []
            temp.append(y)
            temp.append(x)
            temp.append(cropped)
            
            characters.append(temp)
            
            cropped = img[y :y + h , x+ int(w/2) : x + w].copy()#img[y :y + h , x : x + w]
            
            
#            cv2.rectangle(img, (x, y), (x + int(w/2), y + h), (255, 0, 255), 2)
#            cv2.rectangle(img, (x + int(w/2), y), (x + w, y + h), (255, 0, 255), 2)
                #cv2.imshow('color',cropped)
    #        s = 'C:\\Users\\Cristian\\Desktop\\forlearn\\characters\\'+str(y)+'_'+str(x)+'.jpg' 
            iteration=iteration+1
            temp = []
            temp.append(y)
            temp.append(x+int(w/2))
            temp.append(cropped)
            
            characters.append(temp)
        else:
            cropped = img[y :y + h , x : x + w].copy()#img[y :y + h , x : x + w]
#            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                #cv2.imshow('color',cropped)
    #        s = 'C:\\Users\\Cristian\\Desktop\\forlearn\\characters\\'+str(y)+'_'+str(x)+'.jpg' 
            iteration=iteration+1
            temp = []
            temp.append(y)
            temp.append(x)
            temp.append(cropped)
            
            characters.append(temp)
    
#        cv2.imwrite(s , cropped)
#        s = 'C:\\Users\\Cristian\\Desktop\\forlearn\\'+str(y)+'_'+str(x)+'.jpg' 
#     
#        cv2.imwrite(s , cropped)
    cv2.imwrite('C:\\Users\\Cristian\\Desktop\\XXX.jpg',img)


def sort_chars():
    global characters
    global charactersCopy
    
    
    blackim = cv2.imread('C:\\Users\\Cristian\\Desktop\\XXX.jpg') 
    ret,blackim = cv2.threshold(blackim,0,0,cv2.THRESH_BINARY)
    
    
    previousName = ""
    immediatePreviousName = ""
#    indexIteration = 0
    
    characters.sort(key = operator.itemgetter(0,1))
    for index in range(0,len(characters)):
        image = characters[index]
        if (previousName != ""):
            yCurent=image[0]
#            xCurent_dot = image[1]
            if abs(int(immediatePreviousName)-int(yCurent)) < 10:
                immediatePreviousName = yCurent
                yCurent = previousName
            else:
                previousName = yCurent
                immediatePreviousName = previousName
            image[0] = int(yCurent)
        else:
            yCurent=image[0]
#            xCurent_dot = image[1]
            previousName = yCurent
            immediatePreviousName = yCurent
    
    
    
    characters.sort(key = operator.itemgetter(0,1))
    charactersCopy = []
#    for index in range(0,len(characters)):
#        image = characters[index]
#        charactersCopy[0] = image[2].shape[0]
#        charactersCopy[1] = image[2].shape[1]
#    
    for index in range(0,len(characters)):
        temporar = characters[index]
#        print(len(characters[index]))
        
        charactersCopy.append([temporar[2].shape[0],temporar[2].shape[1]])
        blackim[int(temporar[0]):int(temporar[0])+temporar[2].shape[0],int(temporar[1]):int(temporar[1])+temporar[2].shape[1]] = temporar[2].copy()
        temporar=[]
    
    cv2.imwrite('C:\\Users\\Cristian\\Desktop\\XXX.jpg',blackim)
def myFunc(e):
    return (int(e[0]))
def enhance():
    global characters
    global average_w
    for image in characters:
#        temp = image[2].copy()
        gray_image = cv2.cvtColor(image[2], cv2.IMREAD_ANYCOLOR)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
#        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1))
#        
#        height, width, channels = gray_image.shape
#        print(str(width)+'_'+str(average_w))
#        if average_w*10/100 <= width <= average_w*70/100:
#            gray_image = cv2.erode(gray_image, kernel2, iterations=5)    #kernel1 = np.ones((3,3),np.uint8)
            
        gray_image = cv2.erode(gray_image, kernel, iterations=1)    #kernel1 = np.ones((3,3),np.uint8)
        gray_image = cv2.dilate(gray_image, kernel, iterations=1)    #kernel1 = np.ones((3,3),np.uint8)
        
        difference = 20 - gray_image.shape[1]
        if difference > 0 :
            if difference % 2 == 0:
                gray_image=cv2.copyMakeBorder(gray_image, top=0, bottom=0, left=int(difference/2), right=int(difference/2), borderType = cv2.BORDER_CONSTANT, value=[0,0,0])
            else:
                lft = int(difference/2)
                rght = difference - lft
                gray_image=cv2.copyMakeBorder(gray_image, top=0, bottom=0, left=lft, right=rght, borderType = cv2.BORDER_CONSTANT, value=[0,0,0])
            
        gray_image = cv2.resize(gray_image, (20, 20),interpolation=cv2.INTER_AREA)
        
        mean= 0
        
        bordersize=4
        gray_image=cv2.copyMakeBorder(gray_image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
        image[2] = gray_image
#        cv2.imwrite('C:\\Users\\Cristian\\Desktop\\forlearn\\characters\\'+filename, gray_image)
def recognizeE():
    global characters
    global tosend
#    my_string = pickle.dumps(d3)
#    with open("C:\\Users\\Cristian\\Desktop\\LettersModel.json", "wb") as json_file:
#        json_file.write(my_string)
#    print("Saved model to disk")
    file = open("C:\\Users\\Cristian\\Desktop\\LettersModel.json", 'rb')
    d3 = pickle.load(file)
    file.close()
    
    index = 0
    for tre in characters:
        image = characters[index]
        newImage = np.array(image[2])
        newImage = newImage.flatten()
        newImage = newImage.reshape(28*28, 3)
#            newImage = newImage.astype('float32')
#
#    # Normalize to prevent issues with model
#            newImage /= 255
    # check prediction
        ans3 = predict_for_cv(d3, newImage)
        
        file = open("C:\\Users\\Cristian\\Desktop\\Ebigsmallmany.json", 'rb')
        mlp=pickle.load(file)
        file.close()
        
        file = open("C:\\Users\\Cristian\\Desktop\\00.json", 'rb')
        advanced=pickle.load(file)
        file.close()
#            print(filename+" "+str(letter_count[ans3]), end=' ')
        if str(letter_count[ans3])=="e":
            temp=cv2.cvtColor(image[2], cv2.COLOR_RGB2GRAY)
            (thresh, im_bw) = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            newImage = np.array(im_bw)
            newImage = newImage.reshape(1, 784)
            if mlp.predict(newImage)==1 and advanced.predict(newImage)==0 and index+3<len(characters):
                
                print(str(image[0])+"_"+str(image[1])+"_")
#                print(str(int(characters[index+1][1])))
#                print(str(int(characters[index+2][1])))
#                print(str(int(characters[index+3][1])))
#                print(str(int(characters[index+2][1])-int(characters[index+1][1])-15))
#                print(str(int(characters[index+2][1])-int(characters[index+1][1])+30))
#                print(str(int(characters[index+3][1])-int(characters[index+2][1])))
                                
                if int(characters[index+2][1])-int(characters[index+1][1])-15<=int(characters[index+3][1])-int(characters[index+2][1])<=int(characters[index+2][1])-int(characters[index+1][1])+30:
                    eString="E"+getDigitFor(index+1,index+1)
                    eString+=getDigitFor(index+2,index+1)
                    eString+=getDigitFor(index+3,index+1)
                    print(str(image[0])+"_"+str(image[1])+"_"+eString)
                    if eString not in tosend:
                        tosend.append(eString)
#                if eString=="E332":
#                    print(str(2*(int(characters[index+1][1])-int(characters[index+1][1]))+int(image[1])))
#                    print(str(int(characters[index+3][1])))
        index+=1
def getDigitFor(i1,i2):
#    todo: daca deja exista 1 recunoscut, sa compari cu litera E
    global characters
    global charactersCopy
    if i1==i2:
        averagew = charactersCopy[i2+1][1]+charactersCopy[i2+2][1]
        averagew/=2
        if charactersCopy[i1][1]>=80*averagew/100:
            return getDigit(characters[i1][2])
        else:
            return "1"
    elif i1==i2+1:
        averagew = charactersCopy[i2][1]+charactersCopy[i2+2][1]
        averagew/=2
        if charactersCopy[i1][1]>=80*averagew/100:
            return getDigit(characters[i1][2])
        else:
            return "1"
    else:
        avw = charactersCopy[i2][1]+charactersCopy[i2+1][1]
        avw/=2
        print(str(avw)+' '+str(charactersCopy[i1][1]))
        if charactersCopy[i1][1]>=80*avw/100:
            return getDigit(characters[i1][2])
        else:
            return "1"
        
def getDigit(img):
#    todo: try with recognize on different widths
    global modelsDigits
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret,gray_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gray_image = np.array(gray_image)
    gray_image = gray_image / 255. 
    
    gray_image = gray_image.flatten()
    gray_image = gray_image.reshape(1, 784)
    votes=[0,0,0,0,0,0,0,0,0,0]
    for j in range(0,10):
        accumulator=0
        for l in range(0,len(modelsDigits[j])):
            rez = modelsDigits[j][l].predict(gray_image)[0]
            if rez == 0:
                accumulator+=1
            
    #        print(len(modelsDigits[j]))
    #        if accumulator == len(modelsDigits[j]):
        votes[j] += accumulator
    print(votes)
#    print(votes.index(max(votes)))
    return str(votes.index(max(votes)))
    
soc = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
print(str(host))
port = 12346 
soc.bind(('0.0.0.0', port))        # Bind to the port

soc.listen(5)                 # Now wait for client connection.



modelsDigits=[]
models=[]
for i in range(0,10):
    for k in range(0,10):
        if k!=i:
            file = open("C:\\Users\\Cristian\\Desktop\\mixed\\"+str(i)+str(k)+".json", 'rb')
            models.append(pickle.load(file))
            file.close()
    modelsDigits.append(models)
    models=[]

while True:
    c, addr = soc.accept()     # Establish connection with client.
    print ('Got connection from' + str(addr))
#    c.send('Thank you for connecting'.encode())
    tosend.clear()   
    buf = ''
    numberr=0
    typeBin=0
    temp=b''
    final =  b''
    
    while numberr<4:
        temp = c.recv(1)
        numberr+=1
        final=final+temp
        
        print(temp) 
        buf = buf + str(temp)
    print(final)
    typeBin = struct.unpack('!i', bytes(final))[0]
    
    buf = ''
    numberr=0
    temp=b''
    final =  b''
    
    while numberr<4:
        temp = c.recv(1)
        numberr+=1
        final=final+temp
        
        print(temp) 
        buf = buf + str(temp)
    print(final)
    size = struct.unpack('!i', bytes(final))
    print ("receiving %s bytes" % size)
    received = 0
    with open('C:\\Users\\Cristian\\Desktop\\tst.jpg', 'wb') as img:
        while True:
            data = c.recv(524228)
            received+=len(data)
#            print(received)
            img.write(data)
            if int(received) >= int(size[0]):
                break
        img.close()
    print ('received, yay! '+str(typeBin))
    
    binarize('C:\\Users\\Cristian\\Desktop\\tst.jpg',typeBin)
    segment_chars('C:\\Users\\Cristian\\Desktop\\tst.jpg')
    sort_chars()
    enhance()
    recognizeE()
    
    c.send(struct.pack('!i', len(tosend)))
    for elem in tosend:
        print(str(len(elem.encode('utf-8'))))
        c.send(elem.encode('utf-8'))
#    c.send(struct.pack('!i', len(tosend))
    
    if os.path.exists('C:\\Users\\Cristian\\Desktop\\characters'):
        shutil.rmtree('C:\\Users\\Cristian\\Desktop\\characters')
    os.mkdir('C:\\Users\\Cristian\\Desktop\\characters')
    for index in range(0,len(characters)):
        image = characters[index]
        s = 'C:\\Users\\Cristian\\Desktop\\characters\\'+str(image[0])+'_'+str(image[1])+'.jpg' 
#        print(str(image[0])+'_'+str(image[1]))
        cv2.imwrite(s , image[2])
    
    characters.clear()
    charactersCopy.clear()
    c.close() 