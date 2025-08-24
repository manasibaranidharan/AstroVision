import h5py
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, Input, Model

"""def inspect_hdf5(file):
    with h5py.File(file, 'r') as f:
        print("Datasets inside the file:")
        for key in f.keys():
            print(f" - {key}")

inspect_hdf5("Train/dataset/train.h5")

"""


# Load Dataset

def load_HDF5(file):
    with h5py.File(file, 'r') as f:
        x = f['X'][:]
        y = f['y'][:]
    return x, y

x_train, y_train = load_HDF5("Train/dataset/train.h5") # Train.h5
x_val, y_val = load_HDF5("Train/dataset/val.h5") # val.h5
x_test, y_test = load_HDF5("Train/dataset/test.h5") # test.h5

# Data Augmentation

gendata =  ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)


# model CNNS


def model_a():
    inputs = Input(shape=(64,64,3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    

    return Model(inputs, x)

def model_b():
    inputs = Input(shape=(64,64,3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    

    return Model(inputs, x)

# combining two model 

a_model = model_a()
b_model = model_b()

input_layer = Input(shape=(64,64,3))

# output from models
a_out = a_model(input_layer)
b_out = b_model(input_layer)

# combining two models using concatenate feature 

combined = layers.Concatenate()([a_out, b_out])
x = layers.Dense(128, activation="relu",)(combined)
x = layers.Dropout(0.25)(x)
output = layers.Dense(64, activation="sigmoid")(x)

# final combined model

combine_model = Model(inputs=input_layer, outputs=output)


combine_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
# train model

combine_model.fit(gendata.flow(x_train,y_train,batch_size=32), epochs=108 , validation_data=(x_val,y_val))

# Evaluate

test_loss , test_acc = combine_model.evaluate(x_test,y_test)
print(f"TEST ACCURACY: {test_acc:.2f}")  


# Saving model in keras file

combine_model.save("dataset.keras")