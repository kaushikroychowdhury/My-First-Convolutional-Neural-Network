import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## pre-processing .........................................

IMG_shape = (150,150)
Batch_size = 200
## Data Directories ..........................
IMG_train_dir = './rps/'
IMG_test_dir = './rps-test-set/'

# Image Augmentation

img_gen_val = ImageDataGenerator(rescale=1./255,
                                 validation_split=0.2,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest'
                                 )
test_gen = ImageDataGenerator(rescale=1./255)
train_data_gen = img_gen_val.flow_from_directory(
    directory=IMG_train_dir,
    target_size=IMG_shape,
    class_mode='categorical',
    batch_size=Batch_size,
    subset='training'
)

val_data_gen = img_gen_val.flow_from_directory(
    directory=IMG_train_dir,
    target_size=IMG_shape,
    class_mode='categorical',
    batch_size=Batch_size,
    subset='validation'
)

test_data_gen = test_gen.flow_from_directory(
    directory=IMG_train_dir,
    target_size=IMG_shape,
    class_mode='categorical',
    batch_size=Batch_size
)


### MODEL OUTLINE

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

print(model.summary())

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

max_epoch = 40

history = model.fit(
    train_data_gen,
    epochs=max_epoch,
    validation_data=val_data_gen,
    batch_size=Batch_size,
    verbose=2
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(max_epoch)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
print(plt.show())

test_loss, test_acc = model.evaluate(test_data_gen)
print(test_loss,"    ,     ", test_acc)