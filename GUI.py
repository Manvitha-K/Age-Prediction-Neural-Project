#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import Tkinter
from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
from keras.models import load_model
import numpy
import random
import tkFont
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dropout, Flatten, Dense, Input
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD

import numpy as np
import pandas as pd
import os


top = Tk()
top.title('Age and Gender Classification Neural Network Model')
top.geometry('650x350')
font = tkFont.Font(size=10)

l1=Label(top,text = "The model categorizes images to one of the following 8 age categories", font = font,  fg = 'blue')
l1.grid(row=0,column=0, columnspan = 3)

l2=Label(top,text = " [0-2]  [4-6]  [8-13]  [15-20]  [25-32]  [38-43]  [48-53]  [60+] ", font = font, fg = 'Green',)
l2.grid(row=1,column=0, columnspan = 3)

l3 = Label(top, text="")
l3.grid(row=2,column=0)

canvas = Canvas(top, width=200,height=200, bd=0,bg='white')
canvas.grid(row=3, column=0, columnspan = 2)


def showImg():
    File = tkFileDialog.askopenfilename(title='Open Image') 
    e.set(File)
    
    load = Image.open(e.get())
    w, h = load.size
    
    load = load.resize((200,200))
    imgfile = ImageTk.PhotoImage(load )
    
    canvas.image = imgfile
    canvas.create_image(2,2,anchor='nw',image=imgfile)


e = StringVar()
l3 = Label(top, text="")
l3.grid(row=4,column=0)

submit_button = Button(top, text ='Upload Image', command = showImg)
submit_button.grid(row=6, column=0, sticky = E)


ageLabel = Label(top, text="Predicted Vals:")
ageLabel.grid(row=3, column = 2, sticky=E)



def Predict():
    cwd = os.getcwd()
    img=Image.open(e.get())
    img=img.resize((227, 227))
    imgArray = numpy.array(img)
    imgArray = imgArray.reshape(1, 227 , 227 , 3)
    imgArray = imgArray.astype('float32')
    imgArray /= 255.0
    #model=load_model(cwd + '/hdf/cropped_aligned_40.h5')
    model=load_model(cwd + '/hdf/age8_50_base.h5')
    clsimg=model.predict_classes(imgArray)
    print(clsimg) 
    textvar = clsimg
    myDic = {'[0]': '0-2', '[1]':'4-6', '[2]':'8-13', '[3]':'15-20', '[4]': '25-32', '[5]':'38-43', '[6]':'48-53', '7': '60+'}
    textvar = myDic[str(textvar)]
    ageTxt.delete(0.0, Tkinter.END)
    ageTxt.insert('insert', str(textvar)+'\n')
    ageTxt.update()
     
submit_button = Button(top, text ='Predict', command = Predict)
submit_button.grid(row = 6, column = 1, sticky = W)

ageTxt=Text(top,bd=0, width=15, height =1, font='Fixdsys -12')
ageTxt.grid(row=3, column = 3, sticky=W)
top.mainloop()
