import numpy as np
import pickle
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from funcionesAuxiliares import readData, display

# pip install nolearn , conda install libpython , pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
from nolearn.lasagne import BatchIterator
#pip install scikit-image --upgrade
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import random

from sklearn.utils import shuffle
from skimage import exposure
import warnings
import sys

#np.set_printoptions(threshold=np.nan)

NUM_TRAIN = 0
NUM_CLASSES = 0
IMAGE_SHAPE = (0, 0, 0)
CLASS_TYPES = []

#======================================================================================================
train_file = '../signals_database/traffic-signs-data/trainData.p'  #originally sorted
test_file = '../signals_database/traffic-signs-data/testData.p'
signnames = read_csv(
    "../signals_database/traffic-signs-data/signnames.csv").values[:, 1]

train_flipped_file = '../signals_database/traffic-signs-data/trainFlipped3.p'  #WRONG FLIPPED VERSION!
train_flippedCmp_file = '../signals_database/traffic-signs-data/trainFlippedCmp.p'  # GODD VERSION, NOT WORKED YET
train_extended_file = '../signals_database/traffic-signs-data/trainExtended8.p'
train_processed = '../signals_database/traffic-signs-data/trainProcessed.p'
test_processed = '../signals_database/traffic-signs-data/testProcessed.p'
test_sorted = '../signals_database/traffic-signs-data/testSorted.p'
test_sorted_extened = '../signals_database/traffic-signs-data/testSorted_Extended.p'
test_extened = '../signals_database/traffic-signs-data/testExtended.p'
test_processed_extended = '../signals_database/traffic-signs-data/testExtendedProcessed.p'

#======================================================================================================
# Classes of signs that, when flipped horizontally, should still be classified as the same class
self_flippable_horizontally = np.array(
    [11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
# Classes of signs that, when flipped vertically, should still be classified as the same class
self_flippable_vertically = np.array([1, 5, 12, 15, 17])
# Classes of signs that, when flipped horizontally and then vertically, should still be classified as the same class
self_flippable_both = np.array([32, 40])
# Classes of signs that, when flipped horizontally, would still be meaningful, but should be classified as some other class
cross_flippable = np.array([
    [19, 20],
    [33, 34],
    [36, 37],
    [38, 39],
    [20, 19],
    [34, 33],
    [37, 36],
    [39, 38],
])


#======================================================================================================
# Print iterations progress
def print_progress(iteration, total):
    """
    Call in a loop to create terminal progress bar
    
    Parameters
    ----------
        
    iteration : 
                Current iteration (Int)
    total     : 
                Total iterations (Int)
    """
    str_format = "{0:.0f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(100 * iteration / float(total)))
    bar = '|' * filled_length + '-' * (100 - filled_length)

    sys.stdout.write('\r |%s| %s%%' % (bar, percents)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


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
        pickle.dump(file, f, protocol=2)


def mezclar(X, y):
    print("Shuffle Activated!")
    X, y = shuffle(X, y)
    return X, y


def process_dataset(X, y):
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
    print("Preprocessing dataset with {} examples:".format(X.shape[0]))
    #Convert to grayscale, e.g. single channel Y
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)
    """
    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i])
        print_progress(i + 1, X.shape[0])
    """

    # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
    y_flatten = y
    y = np.eye(NUM_CLASSES)[y]

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1, ))
    return X, y, y_flatten


def flip_extend(X, y):
    """
    Extends existing images dataset by flipping images of some classes. As some images would still belong
    to same class after flipping we extend such classes with flipped images. Images of other would toggle 
    between two classes when flipped, so for those we extend existing datasets as well.
    
    Parameters
    ----------
    X       : ndarray
              Dataset array containing feature examples.
    y       : ndarray, optional, defaults to `None`
              Dataset labels in index form.

    Returns
    -------
    A tuple of X and y.    
    """

    X_extended = np.empty(
        [0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y.dtype)

    for c in range(NUM_CLASSES):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y == c], axis=0)
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(
                X_extended, X[y == c][:, :, ::-1, :], axis=0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(
                X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(
            y_extended,
            np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class vertically and they would still belong to same class(ALL of them for this case)...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(
                X_extended, X[y == c][:, ::-1, :, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(
            y_extended,
            np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(
                X_extended, X[y == c][:, ::-1, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(
            y_extended,
            np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

    return (X_extended, y_extended)


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

        if yb is not None:
            batch_size = Xb.shape[0]
            image_size = Xb.shape[1]

            Xb = self.rotate(Xb, batch_size)
            Xb = self.apply_projection_transform(Xb, batch_size, image_size)

        return Xb, yb

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

    def apply_projection_transform(self, Xb, batch_size, image_size):
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


def extend_balancing_classes(X, y, aug_intensity=0.5, counts=None):
    global NUM_CLASSES
    """
    Extends dataset by duplicating existing images while applying data augmentation pipeline.
    Number of generated examples for each class may be provided in `counts`.
    
    Parameters
    ----------
    X             : ndarray
                    Dataset array containing feature examples.
    y             : ndarray, optional, defaults to `None`
                    Dataset labels in index form.
    aug_intensity :
                    Intensity of augmentation, must be in [0, 1] range.
    counts        :
                    Number of elements for each class.
                    
    Returns
    -------
    A tuple of X and y.    
    """
    num_classes = NUM_CLASSES

    _, class_counts = np.unique(y, return_counts=True)
    max_c = max(class_counts)
    total = max_c * num_classes if counts is None else np.sum(counts)

    X_extended = np.empty(
        [0, X.shape[1], X.shape[2], X.shape[3]], dtype=np.float32)
    y_extended = np.empty([0], dtype=y.dtype)
    print("Extending dataset using augmented data (intensity = {}):".format(
        aug_intensity))

    for c, c_count in zip(range(num_classes), class_counts):
        # How many examples should there be eventually for this class:
        print("In the class ", c, " with ", c_count, " classes")
        max_c = max_c if counts is None else counts[c]
        # First copy existing data for this class
        X_source = (X[y == c] / 255.).astype(np.float32)
        y_source = y[y == c]
        X_extended = np.append(X_extended, X_source, axis=0)

        #print("Start 1st part, bacth size: ", X_source.shape[0], " | rango: ", (max_c // c_count) - 1)
        for i in range((max_c // c_count) - 1):
            batch_iterator = AugmentedSignsBatchIterator(
                batch_size=X_source.shape[0], p=1.0, intensity=aug_intensity)
            for x_batch, _ in batch_iterator(X_source, y_source):
                X_extended = np.append(X_extended, x_batch, axis=0)
                #print_progress(X_extended.shape[0], total)
            #print(X_extended.shape[0])

        #print("Start 2nd part,bacth size: ", max_c % c_count)
        batch_iterator = AugmentedSignsBatchIterator(
            batch_size=max_c % c_count, p=1.0, intensity=aug_intensity)

        for x_batch, _ in batch_iterator(X_source, y_source):
            X_extended = np.append(X_extended, x_batch, axis=0)
            #print_progress(X_extended.shape[0], total)
            break
        # Fill labels for added images set to current class.
        print(X_extended.shape[0])
        nuevo_cant = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((nuevo_cant), c, dtype=int))

    return ((X_extended * 255.).astype(np.uint8), y_extended)


def createFlip(X_train, y_train):
    global NUM_TRAIN
    global IMAGE_SHAPE
    global NUM_CLASSES
    global CLASS_TYPES

    print("Start to flip images...")
    X_train, y_train = flip_extend(X_train, y_train)
    CLASS_TYPES, init_per_class, class_counts = np.unique(
        y_train, return_index=True, return_counts=True)
    NUM_TRAIN = X_train.shape[0]
    IMAGE_SHAPE = X_train[0].shape
    NUM_CLASSES = class_counts.shape[0]
    print("Number of Flipped training examples =", NUM_TRAIN)
    print("Image data shape =", IMAGE_SHAPE)
    print("Number of classes =", NUM_CLASSES)
    return X_train, y_train, class_counts


def createExtendedDS(X_train, y_train, class_counts, num_times,
                     augm_intensity):
    global NUM_TRAIN
    global IMAGE_SHAPE
    global NUM_CLASSES
    global CLASS_TYPES

    X_train, y_train = extend_balancing_classes(
        X_train,
        y_train,
        aug_intensity=augm_intensity,
        counts=class_counts * num_times)
    CLASS_TYPES, init_per_class, class_counts = np.unique(
        y_train, return_index=True, return_counts=True)
    NUM_TRAIN = X_train.shape[0]
    IMAGE_SHAPE = X_train[0].shape
    NUM_CLASSES = class_counts.shape[0]
    print("Number of augmenting and extending training data =", NUM_TRAIN)
    print("Image data shape =", IMAGE_SHAPE)
    print("Number of classes =", NUM_CLASSES)
    return X_train, y_train, class_counts


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


def showFlippledImages():
    X, y, _ = readOriginal(train_file)
    X_original = np.empty(
        [0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    X_result = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    Y_result = np.empty([0], dtype=y.dtype)

    for c in range(NUM_CLASSES):
        """
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_original = np.append(X_original, X[y == c], axis=0)
            X_result = np.append(X_result, X[y == c][:, :, ::-1, :], axis=0)
        """
        #-----------------------------------------
        """
        if c in cross_flippable[:, 0]:
            X_original = np.append(X_original, X[y == c], axis=0)
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_result = np.append(
                X_result, X[y == c][:, :, ::-1, :], axis=0
            )  #change X[y == flip_class] to X[y == c] for visualization purposes
        """
        """
        if c in self_flippable_vertically:
            X_original = np.append(X_original, X[y == c], axis=0)
            # ...Copy their flipped versions into extended array.
            X_result = np.append(X_result, X[y == c][:, ::-1, :, :], axis=0)
        """
        """
        if c in self_flippable_both:
            X_original = np.append(X_original, X[y == c], axis=0)
            # ...Copy their flipped versions into extended array.
            X_result = np.append(X_result, X[y == c][:, ::-1, ::-1, :], axis=0)
        """
        Y_result = np.append(
            Y_result,
            np.full((X_result.shape[0] - Y_result.shape[0]), c, dtype=int))

    plot_flipped_examples(X_original, X_result, Y_result, 1)


def showAugmentSamples():
    x_train, y_train, _ = readOriginal(train_file)
    X_input, y_output = mezclar(x_train, y_train)
    cant_conv = 5
    cant_orig_imgs = 3  #number of images TAKEN AS BASED
    ind = range(0, cant_orig_imgs)

    X_input = (X_input / 255.).astype(np.float32)

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


def plot_some_examples(X_data, y_data, n_examples):
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
            if (IMAGE_SHAPE == (32, 32, 1)):
                IMAGE_SHAPE = (32, 32)
            axis.imshow(
                X_data[random_indices[i]].reshape((IMAGE_SHAPE)), cmap='gray')
        print(
            "--------------------------------------------------------------------------------------\n"
        )
        plt.show()
        if c == 5:
            break


def plot_histogram(titulo, class_indices, class_counts):
    # Plot the histogram
    plt.xlabel('Image Type')
    plt.ylabel('Number of Images')
    plt.rcParams["figure.figsize"] = [30, 5]
    axes = plt.gca()
    axes.set_xlim([-1, 43])

    plt.bar(
        class_indices,
        class_counts,
        tick_label=class_indices,
        color='g',
        width=0.8,
        align='center')
    plt.title(titulo)
    plt.show()


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
    ax.set_xlabel('Image type')
    ax.set_xticks(CLASS_TYPES + width / 2)
    ax.set_xticklabels(CLASS_TYPES)
    plt.title(titulo)
    plt.show()
    #plt.bar(CLASS_TYPES, class_counts, tick_label=CLASS_TYPES, width=0.8, align='center',  color=color)


def showHistogram(file, title):
    X_train, y_train, class_counts1 = readOriginal(file)

    plot_histogram(
        title, CLASS_TYPES,
        class_counts1)  #Get initial39209.png and initialTest12630.png


def doFlip():
    X_train, y_train, class_counts1 = readOriginal(train_file)
    X_train, y_train, class_counts2 = createFlip(X_train, y_train)

    #plot_histograms('Class Distribution Original Training data vs New Flipped Training Data', CLASS_TYPES,class_counts1,class_counts2,'b','r')# get flippedImg_59698.png
    new_data = {'features': X_train, 'labels': y_train}
    save_data(new_data, train_flippedCmp_file)


def doExtended():
    X_train, y_train, class_counts1 = readOriginal(train_flipped_file)

    X_train_extended, y_train_extended, class_counts2 = createExtendedDS(
        X_train, y_train, class_counts1, 8, 0.75)

    #plot_histograms('Class Distribution  New Flipped Training Data vs New Extended Training Data', CLASS_TYPES, class_counts1, class_counts2,'b','r')# get ExtendedImg_313672.png
    new_data = {'features': X_train_extended, 'labels': y_train_extended}
    save_data(new_data, train_extended_file)


def sortTestFile():
    X_test, y_test, class_counts1 = readOriginal(test_file)
    #To generate Augment Data we need to sort the test files
    X_sorted, y_sorted = ordenar(X_test, y_test, class_counts1)

    new_data = {'features': X_sorted, 'labels': y_sorted}
    save_data(new_data, test_sorted)


def extendSortedTest():
    X_sorted, y_sorted, class_counts1 = readOriginal(test_sorted)
    X_test_sorted_extended, y_test_sorted_extended, _ = createExtendedDS(
        X_sorted, y_sorted, class_counts1, 5, 0.75)

    new_data = {
        'features': X_test_sorted_extended,
        'labels': y_test_sorted_extended
    }
    save_data(new_data, test_sorted_extened)


def shuffleExtendedSortedTestFile():
    X_test_sorted_extended, y_test_sorted_extended, class_counts1 = readOriginal(
        test_sorted_extened)
    #plot_some_examples(X_test_sorted_extended, y_test_sorted_extended,3)
    X_test_extended, y_test_extended = mezclar(X_test_sorted_extended,
                                               y_test_sorted_extended)
    new_data = {'features': X_test_extended, 'labels': y_test_extended}
    save_data(new_data, test_extened)


def showProcessResult():
    X, y, _ = readOriginal(train_processed)
    plot_some_examples(X, y, 3)
    #img = X[10].reshape(32, 32)
    #plt.axis('off')
    #plt.imshow(img, cmap='binary')
    #plt.show()


if __name__ == "__main__":
    print("Finish importing packages")
    #---------------------------PLOTTING HISTOGRAMS--------------------------------------------------
    #showHistogram(train_file,'Class Distribution on Test Data')
    #plot_some_examples(X_train, y_train,3)

    #-----------------------------------------------------------------------------
    # Prepare a dataset with flipped classes
    #doFlip()

    #-----------------------------------------------------------------------------
    # Prepare a dataset with extended classes
    #doExtended()

    #-----------------------------------------------------------------------------

    #imagenes_entrenam, clases_entrenam, clases_entrenam_flat = process_dataset(X_train,y_train)

    #X_test, y_test, class_counts2 = readOriginal(test_file)
    #X_test,y_test = mezclar(X_test,y_test)
    #imagenes_eval, clases_eval, clases_eval_flat = process_dataset(X_test,y_test)
    #showProcessResult()

    #plot_histograms(
    #    'Class Distribution between Test Data and New Training Data',
    #    CLASS_TYPES, class_counts2, class_counts1,'b','r')  #get finalVS.png

    #new_data = {'features': imagenes_entrenam, 'labels': clases_entrenam_flat}
    #save_data(new_data, train_processed)
    #new_data = {'features': imagenes_eval, 'labels': clases_eval_flat}
    #save_data(new_data, test_processed)

    #-----------------------------------------------------------------------------
    #sortTestFile()

    #-----------------------------------------------------------------------------
    #extendSortedTest()
    #-----------------------------------------------------------------------------

    #shuffleExtendedSortedTestFile()

    #-----------------------------------------------------------------------------
    #X_test_extended, y_test_extended, class_counts1 = readOriginal(
    #    test_extened)
    #imagenes_eval, clases_eval, clases_eval_flat = process_dataset(
    #    X_test_extended, y_test_extended)
    #new_data = {'features': imagenes_eval, 'labels': clases_eval_flat}
    #save_data(new_data, test_processed_extended)

    #X_test, y_test, class_counts1 = readOriginal(test_file)
    #X_test_extended, y_test_extended, class_counts2 = readOriginal(
    #    test_processed_extended)

    #plot_histograms('Class Distribution across New Extended Test Data',
    #                CLASS_TYPES, class_counts1, class_counts2, 'g', 'm')
    #-----------------------------------------------------------------------------
    #showAugmentSamples()
    #-----------------------------------------------------------------------------
    #showFlippledImages()
    #-----------------------------------------------------------------------------
    # +++++++++++ DISCOVER BUG IN FLIPPED FILE
    """
    X_test, y_test, class_counts1 = readOriginal(train_flipped_file)
    X_test_extended, y_test_extended, class_counts2 = readOriginal(
        train_flippedCmp_file)

    plot_histograms('Class Distribution across New Extended Test Data',
                    CLASS_TYPES, class_counts1, class_counts2, 'g', 'm')
    """
    