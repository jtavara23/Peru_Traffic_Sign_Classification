# -*- coding: utf-8 -*-
#-------------------------------------
# Intefarce program to recognize German and Peru Traffic Signs
#-------------------------------------
import io
import sys
import os
import string

#from Tkinter import *
from tkinter import *

#import tkFont
from tkinter import font

from PIL import ImageTk, Image

#from tkFileDialog import askopenfilename
from tkinter import filedialog

#import tkMessageBox as messagebox
from tkinter import messagebox

import analyseSignal as anaSig
filePath = ""
pastfilePath = ""
clean = False
modelType = ""

def on_closing():
    if messagebox.askokcancel("Salir", "Desea salir?"):
        root.destroy()


def updateImage():
    global filePath
    global pastfilePath
    imgIsSelected = True

    if (filePath == pastfilePath):
        # don't override the built-in file class
        filePath = filedialog.askopenfilename()
        if not filePath:
            imgIsSelected = False
        else:
            print("Imagen: " + filePath)
            pathText.delete('1.0', END)
            #queryText.delete('1.0', END)
            # pathText -> just for the view
            pathText.insert(INSERT, ".." + filePath[-28:])
    else:
        print("Nueva ruta: ", filePath)
    pastfilePath = filePath

    if imgIsSelected:
        image = Image.open(filePath)
        image = image.resize((320, 320), Image.ANTIALIAS)  # is (width,height)
        img = ImageTk.PhotoImage(image)
        panel = Label(root, image=img, background='black')
        panel.place(x=10, y=200)
        root.mainloop()


def showImgProc():

    # print processed_images
    lista_imagenes = []
    lista_imagenes.append(Image.open(anaSig.PROCESSED_IMAGES_PATH + "a.jpg"))
    lista_imagenes.append(Image.open(anaSig.PROCESSED_IMAGES_PATH + "b.jpg"))
    #lista_imagenes.append(Image.open(anaSig.PROCESSED_IMAGES_PATH + "c.jpg"))
    lista_imagenes.append(Image.open(anaSig.PROCESSED_IMAGES_PATH + "d.jpg"))
    lista_imagenes.append(Image.open(anaSig.PROCESSED_IMAGES_PATH + "e.jpg"))

    new_h = 170
    new_w = 210
    pos = 25
    COLUMNS = 1

    for ii in range(0, len(lista_imagenes)):
        r, c = divmod(ii, COLUMNS)
        im = lista_imagenes[ii]
        resized = im.resize((new_w, new_h), Image.ANTIALIAS)
        tkimage = ImageTk.PhotoImage(resized)
        myvar = Label(root, image=tkimage)
        myvar.image = tkimage
        myvar.grid(row=r, column=c, padx=650)

    # win.mainloop()

def markPeru():
    global modelType
    modelType = "Peru"
    updateImage()

def markAlem():
    global modelType
    modelType = "Alemania"
    updateImage()

def goQuery():
    global filePath
    global signalRecognized
    global clean
    global res
    global modelType

    print("Ejecutando modelo convolucional")
    signalRecognized = anaSig.runAnalyzer(filePath, modelType)

    showImgProc()  #should be at the end? need to be tested in Windows

    if (clean):
        res.place(y=-80)
    else:
        clean = True
    senWord = Label(
        root,
        text="►► Señal :",
        font=times4,
        background='white')
    senWord.place(x=270, y=720)
    res = Label(
        root,
        text=signalRecognized,
        font=times5,
        background='white',
        fg='blue')
    res.place(x=400, y=720)


# ---------------------------------------------------------------
if __name__ == "__main__":

    root = Tk()
    root.resizable(0, 0)
    root.title('Reconocedor de Señales de Tránsito')
    w = 910
    h = 750
    x = 150
    y = 30

    signalRecognized = ""
    root.geometry("%dx%d+%d+%d" % (w, h, x, y))

    times1 = font.Font(family='Times', size=13, weight='bold')
    times2 = font.Font(family='Helvetica', size=10, weight='bold')
    times3 = font.Font(family='Times', size=12, weight='bold', slant='italic')
    times4 = font.Font(family='Helvetica', size=12, weight='bold')
    times5 = font.Font(family='Helvetica', size=13, weight='bold', slant='italic')
    times5.configure(underline=True)

    la = Label(root, text="Ruta de Imagen: ", font=times3, background='white')
    la.place(x=10, y=70)

    #---------------------------
    peru = Button(root, text='Peru', command=lambda : markPeru())
    peru.place(x=140, y=70)
    alem = Button(root, text='Alemania', command=lambda : markAlem())
    alem.place(x=180, y=70)
    """
    bu = Button(
        root,
        text='Cargar Imagen',
        borderwidth=1,
        command=updateImage,
        highlightbackground='black')
    bu.place(x=240, y=30)"""
    #---------------------------
    pathText = Text(root, height=1, width=40)
    pathText.place(x=10, y=110)

    la = Label(root, text="Imagen:", font=times3, background='white')
    la.place(x=10, y=175)

    imageVer = Image.open("imagenes/fondo.jpg")
    imageHor = imageVer
    imageVer = imageVer.resize((4, h-50), Image.ANTIALIAS)  # is (height, width)
    imageHor = imageHor.resize((w, 3), Image.ANTIALIAS)  # is (height, width)
    img = ImageTk.PhotoImage(imageVer)
    li = Label(root, image=img, background='black')
    li.place(x=380, y=0)
    img = ImageTk.PhotoImage(imageHor)
    li = Label(root, image=img, background='black')
    li.place(x=0, y= h-50)

    # -------RIGHT COLUMN-----------

    la2 = Label(root, text="Tamaño Reducido ►", font=times2, background='white')
    la2.place(x=460, y=70)

    la3 = Label(
        root, text="Espacio BGR para RGB ►", font=times2, background='white')
    la3.place(x=440, y=235)

    la4 = Label(
        root, text="Equalizacion de Histograma ►", font=times2, background='white')
    la4.place(x=410, y=430)

    la5 = Label(
        root, text="Escala de Grises ►", font=times2, background='white')
    la5.place(x=470, y=590)

    #------------------
    res = Label(root, text="Señal: ►►")
    bu = Button(
        root,
        text='Reconocer Señal',
        borderwidth=5,
        highlightbackground='black',
        command=lambda : goQuery(),
        font=times1)
    bu.place(x=85, y=585)
    print("-----------")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.configure(background='white')
    root.mainloop()
