import numpy as np
np.set_printoptions(threshold=np.nan)
import pickle
import cv2
import matplotlib.pyplot as plt
import csv

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,4):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        ind = 0
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
            ind +=1
        gtFile.close()
        print ("Leyendo ",c, " grupo: ",ind)
    return images, labels

def dumpNow(images,labels,name):

    for x in range(0,len(images)):
        images[x] = cv2.resize(images[x], (32,32))

    features = np.array(images)
    labels = np.array(labels)

    #fileObject = open("traffic-signs-data/"+name,'wb')
    fileObject = open(name,'wb')
    info = {'features': features, 'labels': labels}
    pickle.dump(info, fileObject, protocol=2)

if __name__ == "__main__":
    trainImages, trainLabels = readTrafficSigns('../GTSRB/Final_Training')

    dumpNow(trainImages,trainLabels,"train.p")

    #testImages, testLabels = readTrafficSigns('GTSRB/Final_Test')
    #dumpNow(testImages,testLabels,"test.p")

    #print (features[0])
    #print (labels[0])
    #plt.imshow(features[0])
    #plt.show()

