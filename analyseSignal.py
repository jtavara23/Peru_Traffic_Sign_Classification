import numpy as np
import pandas as pd 
import cv2
import pickle
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import tensorflow as tf
from funcionesAuxiliares import readData, plot_example_errors,plot_confusion_matrix
import math
import os
#from keras.backend import manual_variable_initialization as ke

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#np.set_printoptions(threshold=np.nan)

BATCH_SIZE = 421#421
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
rutaDeModelo = 'D:/signalsWindows/models5/'
NUM_TEST = 0


def read_image(imagen):
	
	data = []
	yy, xx = imagen.shape[:2]

	for x in xrange(0,xx):
		for y in xrange(0,yy):
			data.append(imagen[x,y])
	#outFile.write(repr(data)+ "\n")
	return np.matrix(data)

def procesarIMG():
	name = "../signals_database/process/stop2.png"
	img = cv2.imread(name)

	img = cv2.resize(img,(32,32))
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	
	img2 = img

	img = 0.299 * img[:, :, 0] + 0.587 * img[:, :,  1] + 0.114 * img[:, :,  2]
	#Scale features to be in [0, 1]
	#plt.imshow(img,cmap="gray")
	#plt.show()
	#cv2.imwrite('process/gray.jpg',img)
	np.reshape(img, (1,32, 32))
	img = (img / 255.).astype(np.float32) 
	return img

if __name__ == "__main__":
	#x,y = analyse_image('prueba.jpg')
	#"""
	print ("Running session")
	with tf.Session() as sess:
		# Restore latest checkpoint
		sess.run(tf.global_variables_initializer())
		saver = tf.train.import_meta_graph(rutaDeModelo + 'model-49296.meta')
		ult_pto_control = tf.train.latest_checkpoint(rutaDeModelo + '.')
		saver.restore(sess,ult_pto_control )
		print ("Modelo restaurado " + ult_pto_control)
		
		predictor = tf.get_collection("predictor")[0]
		probabilidad = tf.get_collection("acuracia")[0]

		img = procesarIMG()
		#np.set_printoptions(threshold=np.nan)
		
		print (img.shape)
		img = img[np.newaxis,:,:,np.newaxis]
		print (img.shape)
		#print img
		#"""
		#------------FIN DE PROCESAMIENTO DE IMAGEN------------
		results = [0] * 43
		feed_dictx = {NOMBRE_TENSOR_ENTRADA+":0": img, NOMBRE_PROBABILIDAD+":0":False}

		# Calcula la clase usando el predictor de nuestro modelo
		label_pred = sess.run(predictor, feed_dict=feed_dictx)
		print ("Signal: ",label_pred)
		#cualquiera es igual
		#acc= probabilidad.eval(feed_dict = feed_dictx)
		acc = sess.run(probabilidad, feed_dict=feed_dictx)
		print (acc[0][label_pred])
		print (acc)