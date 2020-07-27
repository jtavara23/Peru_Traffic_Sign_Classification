"""
-------------------------------------
 Augment Process for Peru Signs
 It uses 4 from 5 Image Techniques(No flipping)
-------------------------------------
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from util.common_functions import readData, display

# pip install nolearn , conda install libpython , pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
from nolearn.lasagne import BatchIterator
#pip install scikit-image --upgrade
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from skimage.transform import AffineTransform
from skimage import filters
from skimage import exposure
import random

from sklearn.utils import shuffle
import warnings
import sys
#pip install tqdm
from tqdm import tqdm
from tqdm import trange
from sklearn.model_selection import train_test_split
#np.set_printoptions(threshold=np.nan)

NUM_TRAIN = 0
NUM_CLASSES = 0
IMAGE_SHAPE = (0, 0, 0)
CLASS_TYPES = []

#======================================================================================================
#Taken 1st from readTrafficSigns.py then convertPickle.py
train_file = '../signals_database/peru-signs-data/pickleFiles/trainData.p'  #originally sorted
test_file = '../signals_database/peru-signs-data/pickleFiles/testData.p'
signnames = read_csv(
    "../signals_database/peru-signs-data/pickleFiles/senalnames.csv").values[:, 1]

train_normalized_file = '../signals_database/peru-signs-data/pickleFiles/train_1Normalized.p'  #just with clahe and scale [0,1]
train_extended_file = '../signals_database/peru-signs-data/pickleFiles/train_3Extended50.p' #698918 images
train_processed_file = '../signals_database/peru-signs-data/pickleFiles/train_4Processed50.p' # #unsorted and ready for trainingProcess
train_processed_file_split = '../signals_database/peru-signs-data/pickleFiles/train5_split_50.p'

validation_test_file_split = '../signals_database/peru-signs-data/pickleFiles/validation_test_5_split_50.p'
validation_split_file = '../signals_database/peru-signs-data/pickleFiles/validation5_split_50.p'
test_processed_file = '../signals_database/peru-signs-data/pickleFiles/test_1Processed.p'  #sorted and ready for testing [not augmented]

#======================================================================================================


def ordenar(X_data, y_data, class_counts):
    X_extended = np.empty(
        [0, X_data.shape[1], X_data.shape[2], X_data.shape[3]],
        dtype=np.float32)
    y_extended = np.empty([0], dtype=y_data.dtype)

    for c, c_count in zip(range(NUM_CLASSES), class_counts):
        # How many examples should there be eventually for this class:
        print("In the class ", c, " with ", c_count, " images")
        # First copy existing data for this class
        X_source = (X_data[y_data == c])
        y_source = y_data[y_data == c]
        X_extended = np.append(X_extended, X_source, axis=0)
        nuevo_cant = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((nuevo_cant), c, dtype=int))

    return X_extended, y_extended


def mezclar(X, y):
    print("Shuffle Activated!")
    X, y = shuffle(X, y)
    return X, y


def readOriginal(train_file):
    global NUM_TRAIN
    global IMAGE_SHAPE
    global NUM_CLASSES
    global CLASS_TYPES

    X_train, y_train = readData(train_file)

    CLASS_TYPES, init_per_class, class_counts = np.unique(
        y_train, return_index=True, return_counts=True)
    NUM_TRAIN = y_train.shape[0]
    IMAGE_SHAPE = X_train[0].shape
    NUM_CLASSES = class_counts.shape[0]
    print("Number of INITIAL training examples =", NUM_TRAIN)
    print("Image data shape =", IMAGE_SHAPE)
    print("Number of classes =", NUM_CLASSES)
    return X_train, y_train, class_counts


def save_data(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f, protocol=4)


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
class AugmentedSignsBatchIterator(BatchIterator):
    """
    Iterates over dataset in batches.
    Allows images augmentation by randomly rotating, applying projection,
    adjusting gamma, blurring, adding noize and flipping horizontally.
    """

    def __init__(self,
                 batch_size,
                 shuffle=False,
                 seed=42,
                 p=0.5,
                 intensity=0.5):
        """
        Initialises an instance with usual iterating settings, as well as data augmentation coverage
        and augmentation intensity.

        Parameters
        ----------
        batch_size:
                    Size of the iteration batch.
        shuffle   :
                    Flag indicating if we need to shuffle the data.
        seed      :
                    Random seed.
        p         :
                    Probability of augmenting a single example, should be in a range of [0, 1] .
                    Defines data augmentation coverage.
        intensity :
                    Augmentation intensity, should be in a [0, 1] range.

        Returns
        -------
        New batch iterator instance.
        """
        super(AugmentedSignsBatchIterator, self).__init__(
            batch_size, shuffle, seed)
        self.p = p
        self.intensity = intensity

    def transform(self, Xb, yb):
        """
        Applies a pipeline of randomised transformations for data augmentation.
        """
        Xb, yb = super(AugmentedSignsBatchIterator, self).transform(
            Xb if yb is None else Xb.copy(), yb)
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        # Apply 4 transformations in a random way
        if yb is not None:
            batch_size = Xb.shape[0]
            image_size = Xb.shape[1]
            #print("\n\nimage: ",yb)
            #if (random.choice([True, False])):
            #Xb = self.histeq(Xb, batch_size)
            #if (random.choice([True, False])):
            #Xb = self.rotate(Xb, batch_size)
            #if (random.choice([True, False])):
            Xb = self.zoom(Xb, batch_size)
            #else:
            #    Xb = self.projection_transform(Xb, batch_size, image_size)
        #--------------------------------------------------------------------
        #--------------------------------------------------------------------
        return Xb, yb

    ################# image zoom function##############
    def zoom(self, Xb, batch_size):
        image_size = Xb.shape[1]
        indices_zoom = np.random.choice(
            batch_size, int(batch_size * self.p * 0.5), replace=False)
        for k in indices_zoom:
            zoom_fac = self.intensity / (2)
            zoom_x = random.uniform(1 - zoom_fac, 1 + zoom_fac)
            zoom_y = random.uniform(1 - zoom_fac, 1 + zoom_fac)

            transform = AffineTransform(scale=(zoom_x, zoom_y))
            Xb[k] = (warp(
                Xb[k],
                transform,
                output_shape=(image_size, image_size),
                order=1,
                mode='edge'))
        return Xb

    ################# Histogram Equalization ######################
    def histeq(self, Xb, batch_size):
        # Apply histogram equalization on one quarter of the images
        indices_histeq = np.random.choice(
            batch_size, int(batch_size * self.p), replace=False)

        for k in indices_histeq:
            X_rgb = Xb[k]
            X_rgb[:, :, 0] = exposure.equalize_hist(X_rgb[:, :, 0])
            X_rgb[:, :, 1] = exposure.equalize_hist(X_rgb[:, :, 1])
            X_rgb[:, :, 2] = exposure.equalize_hist(X_rgb[:, :, 2])
            Xb[k] = X_rgb

        return Xb

    ########### Image Rotate Function ################
    def rotate(self, Xb, batch_size):
        """
        Applies random rotation in a defined degrees range to a random subset of images.
        Range itself is subject to scaling depending on augmentation intensity.
        """
        for i in np.random.choice(
                batch_size, int(batch_size * self.p), replace=False):
            delta = 30. * self.intensity  # scale by self.intensity
            Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode='edge')
        return Xb

    #######For Affine , Shear, Scale and Rotation, Projective Transform ################
    def projection_transform(self, Xb, batch_size, image_size):
        """
        Applies projection transform to a random subset of images. Projection margins are randomised in a range
        depending on the size of the image. Range itself is subject to scaling depending on augmentation intensity.
        """
        d = image_size * 0.3 * self.intensity
        for i in np.random.choice(
                batch_size, int(batch_size * self.p), replace=False):
            # Top left corner, top margin
            tl_top = random.uniform(-d, d)
            # Top left corner, left margin
            tl_left = random.uniform(-d, d)
            # Bottom left corner, bottom margin
            bl_bottom = random.uniform(-d, d)
            # Bottom left corner, left margin
            bl_left = random.uniform(-d, d)
            # Top right corner, top margin
            tr_top = random.uniform(-d, d)
            # Top right corner, right margin
            tr_right = random.uniform(-d, d)
            # Bottom right corner, bottom margin
            br_bottom = random.uniform(-d, d)
            # Bottom right corner, right margin
            br_right = random.uniform(-d, d)

            transform = ProjectiveTransform()
            transform.estimate(
                np.array(((tl_left, tl_top), (bl_left, image_size - bl_bottom),
                          (image_size - br_right, image_size - br_bottom),
                          (image_size - tr_right, tr_top))),
                np.array(((0, 0), (0, image_size), (image_size, image_size),
                          (image_size, 0))))
            Xb[i] = warp(
                Xb[i],
                transform,
                output_shape=(image_size, image_size),
                order=1,
                mode='edge')

        return Xb


def extend_balancing_classes(X, y, aug_intensity=0.5, isBalanced=False):
    """
    Extends dataset by duplicating existing images while applying data augmentation pipeline.
    Number of generated examples for each class may be provided in `counts`.

    Parameters
    ----------
    X             : ndarray
                    Dataset array containing feature examples.NEEDS TO BE SORTED
    y             : ndarray, optional, defaults to `None`
                    Dataset labels in index form.
    aug_intensity :
                    Intensity of augmentation, must be in [0, 1] range.
    isBalanced        :
                    to know what would be the augment data factor

    Returns
    -------
    A tuple of X and y.
    """

    class_numbers, class_counts = np.unique(y, return_counts=True)
    max_c = max(class_counts)

    X_extended = np.empty(
        [0, X.shape[1], X.shape[2], X.shape[3]], dtype=np.float32)
    y_extended = np.empty([0], dtype=y.dtype)
    dataLimiter = 0

    print("Extending dataset using augmented data (intensity = {}):".format(
        aug_intensity))

    #for i in trange(len(class_numbers)):
    #    BATCH_SIZE = class_counts[i] - 1

    for c, BATCH_SIZE in zip(range(len(class_numbers)), class_counts):
        #if c == 4:
        #    break
        print("In the class ", c, " with ", BATCH_SIZE, " images")
        X_source = X[dataLimiter:dataLimiter + BATCH_SIZE]
        y_source = y[dataLimiter:dataLimiter + BATCH_SIZE]
        # First copy existing data for this class
        X_extended = np.append(X_extended, X_source, axis=0)
        y_extended = np.append(y_extended, y_source, axis=0)

        if not isBalanced:
            augment_factor = (max_c // BATCH_SIZE) - 1
            remainder = max_c % BATCH_SIZE
        else:
            augment_factor = 50
            remainder = 0

        numb_new_imgs = 0
        print("1st bacth size: ", BATCH_SIZE, ", times: ", augment_factor)
        print("2nd bacth size: ", remainder)

        for i in range(augment_factor):
            batch_iterator = AugmentedSignsBatchIterator(
                batch_size=(BATCH_SIZE), p=1.0, intensity=aug_intensity)
            for x_aug, y_aug in batch_iterator(X_source, y_source):
                numb_new_imgs += x_aug.shape[0]
                X_extended = np.append(X_extended, x_aug, axis=0)
                y_extended = np.append(y_extended, y_aug, axis=0)

        if not isBalanced:
            batch_iterator = AugmentedSignsBatchIterator(
                batch_size=(remainder), p=1.0, intensity=aug_intensity)

            for x_aug, y_aug in batch_iterator(
                    X[dataLimiter:dataLimiter + remainder],
                    y[dataLimiter:dataLimiter + remainder]):
                numb_new_imgs += x_aug.shape[0]
                X_extended = np.append(X_extended, x_aug, axis=0)
                y_extended = np.append(y_extended, y_aug, axis=0)

        print(numb_new_imgs, " were added. Total => ",
              y_source.shape[0] + numb_new_imgs)
        print("---------------------------------------------------")
        dataLimiter += BATCH_SIZE

    return (X_extended, y_extended)


def createExtendedDS(X_data, y_data, class_counts, Balanced, augm_intensity):
    global NUM_TRAIN
    global IMAGE_SHAPE
    global NUM_CLASSES
    global CLASS_TYPES

    X_data, y_data = extend_balancing_classes(
        X_data, y_data, aug_intensity=augm_intensity, isBalanced=Balanced)

    CLASS_TYPES, init_per_class, class_counts = np.unique(
        y_data, return_index=True, return_counts=True)

    NUM_TRAIN = X_data.shape[0]
    IMAGE_SHAPE = X_data[0].shape
    NUM_CLASSES = class_counts.shape[0]
    print("Number of augmenting and extending training data =", NUM_TRAIN)
    print("Image data shape =", IMAGE_SHAPE)
    print("Number of classes =", NUM_CLASSES)
    return X_data, y_data, class_counts


def doExtended(inputFile, balanced, isTraining=True):
    X_data, y_data, class_counts1 = readOriginal(inputFile)

    if not isTraining:
        sorter = np.argsort(y_data)
        y_data = y_data[sorter]
        X_data = X_data[sorter]
    #"""
    X_data_extended, y_data_extended, class_counts2 = createExtendedDS(
        X_data, y_data, class_counts1, balanced, 0.75)

    new_data = {'features': X_data_extended, 'labels': y_data_extended}
    save_data(new_data, train_extended_file)# or could be train_extended_balanced_file(balanced =>False)
    print("Pickle saved")
    plot_histograms(
            'Class Distribution  New Flipped Training Data vs New Extended Training Data',
            CLASS_TYPES, class_counts1, class_counts2, 'b',
            'r')  # get ExtendedImg_313672.png
    #"""


#----------------------------------------------------------------------------------------
#-----------------------------------CLAHE TECHNIQUE--------------------------------------
#----------------------------------------------------------------------------------------


def applyClahe(X):
    X_ = []
    for k in trange(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_rgb = exposure.equalize_adapthist(X[k])  #, clip_limit=0.03
            X_.append(X_rgb)
        #print(k + 1, X_[k])
    return (np.asarray(X_)).astype(np.float32)


def normalizeData(X_data, y_data, class_counts, targetFile):
    X_data = applyClahe(X_data)
    #plot_some_examples(X_data, y_data, 5, 3)
    new_data = {'features': X_data, 'labels': y_data}
    save_data(new_data, targetFile)
    print("Pickle saved.")


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


#At the end of Augment Data
def convertToGrayScale(inputFile, outputFile):
    """
    Performs feature scaling, one-hot encoding of labels and shuffles the data if labels are provided.
    Assumes original dataset is sorted by labels.

    Parameters
    ----------
    X                : ndarray
                       Dataset array containing feature examples.
    y                : ndarray, optional, defaults to `None`
                       Dataset labels in index form.
    Returns
    -------
    A tuple of X and y.
    """
    X, y, _ = readOriginal(inputFile)

    print("Preprocessing dataset with {} examples".format(X.shape[0]))
    #Convert to grayscale, e.g. single channel Y
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    #NOT NECESSARY HERE BECAUSE IT WAS PERFORM ON CLAHE METHOD
    #X = (X / 255.).astype(np.float32)

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1, ))

    new_data = {'features': X, 'labels': y}

    save_data(new_data, outputFile)


#----------------------------------------------------------------------------------------
#---------------------------PLOT/SHOW IMAGES AND  HISTOGRAMS-----------------------------
#----------------------------------------------------------------------------------------


def plot_flipped_examples(X_origin, X_data, y_data, n_examples):
    #[X_data] para mostrar data y [y_data] para analizar indices
    #""" data NEEDS TO BE SORTED in order to work properly!!!!
    global IMAGE_SHAPE
    CLASS_TYPES, init_per_class, class_counts = np.unique(
        y_data, return_index=True, return_counts=True)
    col_width = max(len(name) for name in signnames)

    rows = len(CLASS_TYPES)
    columns = n_examples + 1

    fig = plt.figure(figsize=(rows, columns))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    pos = 0
    for c, c_index_init, c_count in zip(CLASS_TYPES, init_per_class,
                                        class_counts):
        print(c, " ", c_index_init, " to ", c_index_init + c_count)
        print("Class %i: %-*s  %s samples" % (c, col_width, signnames[c],
                                              str(c_count)))
        random_indices = random.sample(
            range(c_index_init, c_index_init + c_count), n_examples)
        print("Chosen:", random_indices)

        axis = fig.add_subplot(rows, columns, pos + 1, xticks=[], yticks=[])
        axis.imshow(X_origin[random_indices[0]].reshape((IMAGE_SHAPE)))
        axis = fig.add_subplot(rows, columns, pos + 2, xticks=[], yticks=[])
        axis.imshow(X_data[random_indices[0]].reshape((IMAGE_SHAPE)))
        pos += 2
        print(
            "--------------------------------------------------------------------------------------\n"
        )

    plt.show()



def showAugmentSamples(file):
    #WORKS WITH SCALE IMAGES(pixels between 0 & 1)
    X_input, y_output, cc = readOriginal(file)
    #X_input, y_output = ordenar(X_input, y_output, cc)
    #X_input = (X_input / 255)

    cant_conv = 5
    cant_orig_imgs = 6  #number of images TAKEN AS BASED
    #ind = range(0, cant_orig_imgs)
    ind = random.sample(range(0, NUM_TRAIN), cant_orig_imgs)
    #print(signnames[ind])
    fig = plt.figure(figsize=(cant_orig_imgs, cant_conv + 1))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    #plot imgs in a vertical way
    more = False

    for k in range(cant_conv):
        batch_iterator = AugmentedSignsBatchIterator(
            batch_size=cant_orig_imgs, p=1.0, intensity=0.75)
        for x_batch, y_batch in batch_iterator(X_input[ind], y_output[ind]):
            j = k + 1
            for i in range(cant_orig_imgs):
                if more == False:
                    axis = fig.add_subplot(
                        cant_orig_imgs,
                        cant_conv + 1,
                        (i + j),  #(i+j) means start pos. of new row
                        xticks=[],
                        yticks=[])
                    axis.imshow(X_input[ind[i]].reshape((IMAGE_SHAPE)))
                j += 1
                axis = fig.add_subplot(
                    cant_orig_imgs,
                    cant_conv + 1, (i + j),
                    xticks=[],
                    yticks=[])
                axis.imshow(x_batch[i].reshape((IMAGE_SHAPE)))
                j += cant_conv - 1  #2
            break
        more = True
    plt.show()


def plot_some_examples(X_data, y_data, n_examples, breakAt=7):
    #[X_data] para mostrar data y [y_data] para analizar indices
    #""" data NEEDS TO BE SORTED in order to work properly!!!!
    global IMAGE_SHAPE
    CLASS_TYPES, init_per_class, class_counts = np.unique(
        y_data, return_index=True, return_counts=True)
    col_width = max(len(name) for name in signnames)

    for c, c_index_init, c_count in zip(CLASS_TYPES, init_per_class,
                                        class_counts):
        print(c, " ", c_index_init, " to ", c_index_init + c_count)
        print("Class %i: %-*s  %s samples" % (c, col_width, signnames[c],
                                              str(c_count)))
        fig = plt.figure(figsize=(8, 1))
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        random_indices = random.sample(
            range(c_index_init, c_index_init + c_count), n_examples)
        print(random_indices)
        for i in range(n_examples):
            axis = fig.add_subplot(1, n_examples, i + 1, xticks=[], yticks=[])
            if (X_data[0].shape == (60, 60, 1)):
                IMAGE_SHAPE = (60, 60)
            axis.imshow(
                X_data[random_indices[i]].reshape((IMAGE_SHAPE)), cmap='gray')
        print(
            "--------------------------------------------------------------------------------------\n"
        )
        plt.show()
        if c == breakAt:
            break


def plot_histograms(titulo, CLASS_TYPES, class_counts1, class_counts2, color1,
                    color2):
    #Plot the histogram
    #plt.xlabel('Clase')
    #plt.ylabel('Numero de imagenes')
    #plt.rcParams["figure.figsize"] = [30, 5]
    #axes = plt.gca()
    #axes.set_xlim([-1,43])

    #Calculate optimal width
    width = np.min(np.diff(CLASS_TYPES)) / 3
    width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(CLASS_TYPES, class_counts1, width, color=color1, label='-Ymin')
    ax.bar(
        CLASS_TYPES + width, class_counts2, width, color=color2, label='Ymax')
    ax.set_xlabel('Categorias')
    ax.set_xticks(CLASS_TYPES + width / 2)
    ax.set_xticklabels(signnames,rotation=25)
    plt.title(titulo)
    plt.show()
    #plt.bar(CLASS_TYPES, class_counts, tick_label=CLASS_TYPES, width=0.8, align='center',  color=color)


def showHistogram(file, title):
    X_train, y_train, class_counts = readOriginal(file)
    # Plot the histogram
    plt.xlabel('Categorias')
    plt.xticks( fontsize=6,rotation=25)
    plt.ylabel('Numero de Imagenes')
    plt.rcParams["figure.figsize"] = [30, 5]
    axes = plt.gca()
    axes.set_xlim([-1, 7])

    plt.bar(
        CLASS_TYPES,
        class_counts,
        tick_label=signnames,
        color='b',
        width=0.7,
        align='center')
    plt.title(title)
    plt.show()

def saveSplitData(X_data,y_data,targetFile):
    new_data = {'features': X_data, 'labels': y_data}
    save_data(new_data, targetFile)
    print("Pickle saved.")


if __name__ == "__main__":
    print("Finish importing packages")

    #X_train, y_train, _ = readOriginal(train_file)
    #X_train, y_train, _ = readOriginal(train_processed_balanced_file)
    #showHistogram(train_file,'Distribucion de Clases en Dataset Inicial')
    #plot_some_examples(X_train, y_train,5)

    #X_test, y_test, _ = readOriginal(test_file)
    #showHistogram(X_test,'Distribucion de datos en el Dataset de Evaluación')
    #plot_some_examples(X_test, y_test,5)

    #--------------------NORMALIZE DATA----1st step-----------------------------------------------------
    #x, y, cc = readOriginal(train_file)
    #normalizeData(x, y, cc, train_normalized_file)
    #100%|███████████████████████████████████████████████████████| 614/614 [00:09<00:00, 61.84it/s]
    #x, y, cc = readOriginal(test_file)
    #normalizeData(x, y, cc, test_normalized_file)
    #100%|####################################################| 12630/12630 [08:04<00:00, 26.05it/s]

    #------------------------------------2nd step-----------------------------------------
    # Prepare a dataset with flipped classes
    #Doenst apply for this dataset
    #------------------------------3rd step-----------------------------------------------
    # Prepare a dataset with extended classes
    #doExtended(train_normalized_file, balanced=False)
    #doExtended(
    #    train_normalized_file,
    #    balanced=True)  # BE CAREFUL! WITH OVERWRITTEN THE FILE #este fue usado

    #-------------------------PROCESS FILES 4th step------------------------------------------
    #convertToGrayScale(train_extended_balanced_file,train_processed_balanced_file)#not used
    #convertToGrayScale(train_extended_file,train_processed_file) #Used
    """Used to print Result
    #x, y, cc = readOriginal(train_normalized_file)
    #x , y = convertToGrayScale(x,y)#Final Test File
    #plot_some_examples(x, y, 8);
    """
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    showAugmentSamples(train_normalized_file)
    #-----------------------------------------------------------------------------
    #X_train, y_train, class_counts1 = readOriginal(train_normalized_file)
    #X_train, y_train, class_counts2 = readOriginal(train_extended_file)

    #plot_histograms(
    #    'Distribucion de Clases del Dataset Original vs Dataset Aumentado(x50 veces) ',
    #    CLASS_TYPES, class_counts1, class_counts2, 'g',
    #    'r')
    #
    #DIVIDE VALID_TEST AND TRAIN SETS FROM: train_processed_file
    """
    X_train, y_train = readData(train_processed_file)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    saveSplitData(X_train,y_train,train_processed_file_split)
    saveSplitData(X_validation,y_validation,validation_test_file_split)
    """
    #DIVIDE VALID AND TEST SETS FROM: validation_test_file_split
    """
    X_train, y_train = readData(validation_test_file_split)

    X_valid, X_test, y_valid, y_test = train_test_split(X_train, y_train, test_size=0.60)
    # 40 va para VALIDATION (10% del total)
    # 60 va para TEST (15% del total)
    saveSplitData(X_valid, y_valid, validation_split_file)
    saveSplitData(X_test, y_test, test_processed_file)
    """
    #Read final sets
    #x, y ,_= readOriginal(validation_split_file)
    #xx, yy,_ = readOriginal(test_processed_file)