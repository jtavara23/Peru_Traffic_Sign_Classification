parameters = Parameters(
    # Data parameters
    num_classes = 43,
    image_size = (32, 32),
    # Training parameters
    batch_size = 256,
    max_epochs = 1001,
    log_epoch = 1,
    print_epoch = 1,
    # Optimisations
    learning_rate_decay = False,
    learning_rate = 0.0001,
    l2_reg_enabled = True,
    l2_lambda = 0.0001,
    early_stopping_enabled = True,
    early_stopping_patience = 100,
    resume_training = True,
    # Layers architecture
    conv1_k = 5, conv1_d = 32, conv1_p = 0.9,
    conv2_k = 5, conv2_d = 64, conv2_p = 0.8,
    conv3_k = 5, conv3_d = 128, conv3_p = 0.7,
    fc4_size = 1024, fc4_p = 0.5)

import random
import pickle
from sklearn.cross_validation import train_test_split

train_dataset_file = "traffic-signs-data/train.p"
test_dataset_file = "traffic-signs-data/test.p"
train_extended_dataset_file = "traffic-signs-data/train_extended.p"
train_balanced_dataset_file = "traffic-signs-data/train_balanced.p"

X_train, y_train = load_pickled_data(train_dataset_file, ['features', 'labels'])
print("Number of training examples in initial dataset =", X_train.shape[0])
_, class_counts = np.unique(y_train, return_counts = True)
X_train, y_train = flip_extend(X_train, y_train)
print("Number of training examples after horizontal flipping =", X_train.shape[0])

# Prepare a dataset with balanced classes
X_train_balanced, y_train_balanced = extend_balancing_classes(X_train, y_train, aug_intensity = 0.75, counts = np.full(43, 20000, dtype = int))
print("Number of training examples after augmenting and balancing training data =", X_train_balanced.shape[0])
pickle.dump({
        "features" : X_train_balanced,
        "labels" : y_train_balanced
    }, open(train_balanced_dataset_file, "wb" ) )
print("Balanced dataset saved in", train_balanced_dataset_file)

# Prepare a dataset with extended classes
X_train_extended, y_train_extended = extend_balancing_classes(X_train, y_train, aug_intensity = 0.75, counts = class_counts * 20)
print("Number of training examples after augmenting and extending training data =", X_train_extended.shape[0])
pickle.dump({
        "features" : X_train_extended,
        "labels" : y_train_extended
    }, open(train_extended_dataset_file, "wb" ) )
print("Extended dataset saved in", train_extended_dataset_file)



#Preprocess all datasets:
#In [ ]:

import pickle

train_extended_dataset_file = "../signals_database/traffic-signs-data/train_extended.p"
train_balanced_dataset_file = "../signals_database/traffic-signs-data/train_balanced.p"
train_extended_preprocessed_dataset_file = "../signals_database/traffic-signs-data/train_extended_preprocessed.p"
train_balanced_preprocessed_dataset_file = "../signals_database/traffic-signs-data/train_balanced_preprocessed.p"

test_dataset_file = "traffic-signs-data/test.p"
test_preprocessed_dataset_file = "traffic-signs-data/test_preprocessed.p"

X_train, y_train = load_and_process_data(train_balanced_dataset_file)
pickle.dump({
        "features" : X_train,
        "labels" : y_train
    }, open(train_balanced_preprocessed_dataset_file, "wb" ) )
print("Preprocessed balanced training dataset saved in", train_balanced_preprocessed_dataset_file)

X_train, y_train = load_and_process_data(train_extended_dataset_file)
pickle.dump({
        "features" : X_train,
        "labels" : y_train
    }, open(train_extended_preprocessed_dataset_file, "wb" ) )
print("Preprocessed extended training dataset saved in", train_extended_preprocessed_dataset_file)

X_test, y_test = load_and_process_data(test_dataset_file)
pickle.dump({
        "features" : X_test,
        "labels" : y_test
    }, open(test_preprocessed_dataset_file, "wb" ) )
print("Preprocessed extended testing dataset saved in", test_preprocessed_dataset_file)
