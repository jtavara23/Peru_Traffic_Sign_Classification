#-------------------------------------
# Helper file for Analyzing Plotting
#-------------------------------------
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import pandas as pd
import util.confusion_matrix as confMat
import seaborn as sns
import pickle
"""

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

"""


def readData(file):

    with open(file, mode='rb') as f:
        data = pickle.load(
            f#, encoding='latin1' ... doesnt work in ubuntu
        )  #Pickle incompatability of numpy arrays between Python 2 and 3, latin1 seems to work

    X_file, y_file = data['features'], data['labels']

    return X_file, y_file


# display an image
def display(img, mod, tam, tipo='binary'):

    if (mod):
        img = img.reshape(tam, tam)
    plt.axis('off')
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='binary')
    plt.imshow(img, cmap=tipo)  #deafult was binary
    #plt.imshow(img)
    plt.show()


# Strutified shuffle is used insted of simple shuffle in order to achieve sample balancing
# or equal number of examples in each of 10 classes.
# Since there are different number of examples for each 10 classes in the MNIST data you may
# also use simple shuffle.
def stratified_shuffle(labels, num_classes):
    ix = np.argsort(labels).reshape((num_classes, -1))
    for i in range(len(ix)):
        np.random.shuffle(ix[i])
    return ix.T.reshape((-1))


#Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 16
    img_shape = (28, 28)
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "REAL: {0}".format(cls_true[i])
        else:
            xlabel = "Des: {0}, Calc: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])


#Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred, correct, images, labels_flat):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = labels_flat[incorrect]

    # Plot the first 9 images.
    plot_images(
        images=images[0:16], cls_true=cls_true[0:16], cls_pred=cls_pred[0:16])


def plot_confusion_matrix(cls_pred, cls_true, num_classes):
    # This is called from print_test_accuracy() below.
    signnames = pd.read_csv(
        "../signals_database/traffic-signs-data/signnames.csv").values[:, 1]
    # cls_pred is an array of the predicted classifications for the test-set

    # cls_true is an array of the true classifications for the test-set

    # Get the confusion matrix using sklearn.
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    cm_tot = cm
    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
    # Print the confusion matrix as text.
    #print(cm)

    # Plot the confusion matrix as an image.
    #plt.matshow(cm)
    plt.figure(figsize = (27,27))
    #cmao = Greens,Oranges,Blues, PuBu,Purples,Reds,GnBu
    plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
    plt.title("Matriz de Confusion")
    #plt.colorbar()
    tick_marks = np.arange(num_classes)

    plt.xticks( range(num_classes), tick_marks)#, rotation=90)
    plt.yticks( range(num_classes), tick_marks)
    plt.xticks(fontsize = 6)
    plt.yticks(fontsize = 6)

    thresh = cm.max() / 2.
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        dato = str(cm[i, j]) #+ "\n" + str(cm_tot[i,j])
        plt.text(
            j,
            i,
            dato,
            linewidth=2,
            fontsize=8,
            horizontalalignment="center",
            #bbox=dict(facecolor='red', alpha=0.5),
            color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predecida', fontsize = 24)
    plt.ylabel('Deseada', fontsize = 24)
    plt.tight_layout()
    plt.show()


def plot_conv_weights(w, xs, ys, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    #w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]
    print(num_filters)

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(xs, ys)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(
                img,
                vmin=w_min,
                vmax=w_max,
                interpolation='nearest',
                cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


#def plot_conv_layer(sess, layer, image,xs,ys):
def plot_conv_layer(w, input_channel=0):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    #feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    #values = sess.run(layer, feed_dict=feed_dict)
    #print(w)
    # Number of filters used in the conv. layer.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]
    print(num_filters)
    xs = int(num_filters/8)
    ys = 8

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(xs, ys)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[0, :, :, i]
            #img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest',#'nearest' | sinc
                      cmap='binary'# cmap = 'seismic | binary'
                      )

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_confusion_matrix_Large(cls_pred, cls_true, num_classes):
    confMat._test_data_class( cls_true, cls_pred, num_classes)


def plot_roc2(cls_pred,y_true, classes_to_plot=None, title='ROC Curves',
                   plot_micro=True, plot_macro=True,
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        cls_pred (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".
        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.
        plot_macro (boolean, optional): Plot the macro average ROC curve.
            Defaults to ``True``.
        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> cls_pred = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, cls_pred)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """

    from sklearn.preprocessing import label_binarize
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from scipy import interp

    y_true = np.array(y_true)
    cls_pred = np.array(cls_pred)

    classes = np.unique(y_true)
    probas = cls_pred

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    print("indices_to_plot ",indices_to_plot)
    print("--------------")
    print("probas:" , probas)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i],
                                                pos_label=classes[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                          ''.format(classes[i], roc_auc))

    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='deeppink', linestyle=':', linewidth=4)

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr,
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
def plot_roc(cls_pred, cls_true, num_classes):
    from sklearn.metrics import roc_curve,auc
    from scipy import interp
    from itertools import cycle
    PFP = dict()
    PVP = dict()
    roc_auc = dict()
    avg_PFP = avg_PVP = 0

    pred_array = np.array(pd.get_dummies(cls_pred))
    test_array = np.array(pd.get_dummies(cls_true))

    for i in range(num_classes):
        PFP[i], PVP[i], _ = roc_curve(test_array[:, i], pred_array[:, i])
        roc_auc[i] = auc(PFP[i], PVP[i])
        avg_PFP += PFP[i]
        avg_PVP += PVP[i]
        #print(i," -> ",  PFP[i],"  |||| " ,PVP[i], " |||| ROC Area: ", roc_auc[i])
        #print("---------------------")
    print("avg_PFP: ", (avg_PFP/num_classes)[1] ," -> avg PVN:",1-((avg_PFP/num_classes)[1]),"  |||  avg_PVP: " , (avg_PVP/num_classes)[1] )
    #--------------------------------------------------------------------------
    #Another evaluation measure for multi-class classification is macro-averaging,
    #which gives equal weight to the classification of each label.
    all_fpr = np.unique(np.concatenate([PFP[i] for i in range(num_classes)]))

    #print("all_fpr",all_fpr)
    #[0.00000000e+00 2.45158127e-04 2.49314385e-04 2.59336100e-04 5.45851528e-04 7.72996650e-04 1.00000000e+00]

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, PFP[i], PVP[i])


    mean_tpr /= num_classes
    #print("mean_tpr",mean_tpr)
    #[0.28508217 0.81205758 0.81857359 0.82855115 0.95635019 0.99827774 1.   ]

    PFP["macro"] = all_fpr
    PVP["macro"] = mean_tpr
    roc_auc["macro"] = auc(PFP["macro"], PVP["macro"])
    #--------------------------------------------------------------------------
    lw=2
    plt.figure(figsize=(13,13))
    """
    plt.plot(PFP["macro"], PVP["macro"],
            label='macro-average ROC curve (area = {0:0.3f})'
                ''.format(roc_auc["macro"]),
            color='green', linestyle=':', linewidth=4)
    """
    print('AUC-ROC: {0:0.4f}'
        .format(roc_auc["macro"]))
    lines = cycle(['-', '--', '-.', ':'])
    colors = cycle(['aqua', 'cornflowerblue', 'red', 'darkorange', 'black', 'blue', 'brown', 'green','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color, line in zip(range(num_classes), colors, lines):
        plt.plot(PFP[i], PVP[i], color=color, lw=lw, linestyle=line,
                label='ROC curve para la clase {0} (AUC ROC = {1:0.4f})'
                ''.format(i, roc_auc[i]))
    #"""
    plt.plot([0, 1], [0, 1], 'k--',color='gray', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Suposicion Aleatoria',(.5,.48),color='gray')
    plt.xlabel('Proporción de Falsos Positivos (PFP)',fontsize=12)
    plt.ylabel('Proporción de Verdaderos Positivos (PVP)',fontsize=12)
    plt.title('Curvas ROC - AUC={0:0.4f}'
        .format(roc_auc["macro"]))
    if(num_classes < 10):
        plt.legend(loc="lower right")
    else:
        plt.legend(loc=(1.05,-.02))
    plt.tight_layout()  #set layout slim
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

def plot_pr_curve(y_score, Y_test, num_classes):
     #The measure of quality of precision-recall curve is average precision.
    #This average precision equals the exact area under not-interpolated (that is, piecewise constant) precision-recall curve.
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from funcsigs import signature

    #print(y_score)
    y_score_oneHot = np.eye(num_classes,dtype=int)[y_score]
    Y_test_oneHot = np.eye(num_classes,dtype=int)[Y_test]
    """
    for i in range(num_classes):
        print(Y_test_oneHot[: , i])
        print("----------")
    """
    # For each class
    avg_precision = 0
    precision = dict()
    recall = dict()
    average_precision_recall = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test_oneHot[:,i],
                                                            y_score_oneHot[:,i])
        average_precision_recall[i] = average_precision_score(Y_test_oneHot[:, i], y_score_oneHot[:, i])
        #print(precision[i], " and ", average_precision_recall[i])
        if(len(precision[i]) == 2):
            avg_precision += precision[i][0]
        else:
            avg_precision += precision[i][1]
    #------------------------------------------------------------------------------------------
    # A "micro-average": quantifying score on all classes jointly
    #  A "macro-average": Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    average_precision_recall["micro"] = average_precision_score(Y_test_oneHot, y_score_oneHot,
                                                        average="micro")
    print("--------------------")
    print('macro-promedio en todas las clases, AUC-PR=: {0:0.4f}'
        .format(average_precision_recall["micro"]))
    print('PPV (precision): ',avg_precision/num_classes)
    print("--------------------")
    """
    precision["macro"], recall["macro"], _ = precision_recall_curve(Y_test_oneHot.ravel(),
        y_score_oneHot.ravel())
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                    **step_kwargs)
    """
    #------------------------------------------------------------------------------------------
    plt.figure(figsize=(13,13))

    from itertools import cycle
    lines = cycle(['-', '--', '-.', ':'])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'black', 'blue', 'brown', 'green','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    for i, color, line in zip(range(num_classes), colors, lines):
        plt.plot(recall[i], precision[i], color=color, lw=2, linestyle=line,
        label='Precision-recall para la clase {0} (AUC-PR = {1:0.3f})'
                  ''.format(i, average_precision_recall[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Curvas Precision - Recall, AUC={0:0.4f}'
        .format(average_precision_recall["micro"]))
    if(num_classes < 10):
        plt.legend(loc="lower left")
    else:
        plt.legend(loc=(1.05,-.02))
    plt.tight_layout()  #set layout slim
    #"""
