from unet import build_unet
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import random


large_image_stack = tiff.imread('training.tif')
large_mask_stack = tiff.imread('training_groundtruth.tif')

all_img_patches = []
for img in range(large_image_stack.shape[0]):

    large_image = large_image_stack[img]
    patches_img = patchify(large_image, (256, 256), step=256)  # Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = (single_patch_img.astype('float32')) / 255.
            all_img_patches.append(single_patch_img)

images = np.array(all_img_patches)
images = np.expand_dims(images, -1)

all_mask_patches = []
for img in range(large_mask_stack.shape[0]):

    large_mask = large_mask_stack[img]

    patches_mask = patchify(large_mask, (256, 256), step=256)

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i, j, :, :]
            single_patch_mask = single_patch_mask / 255.

            all_mask_patches.append(single_patch_mask)

masks = np.array(all_mask_patches)
masks = np.expand_dims(masks, -1)

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.25, random_state=0)


image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

IMG_HEIGHT = images.shape[1]
IMG_WIDTH = images.shape[2]
IMG_CHANNELS = images.shape[3]

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = build_unet(input_shape)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

seed = 24

img_data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         shear_range=0.5,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                          width_shift_range=0.3,
                          height_shift_range=0.3,
                          shear_range=0.5,
                          zoom_range=0.3,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect',
                          preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(
                              x.dtype))  # Binarize the output again.

image_data_generator = ImageDataGenerator(**img_data_gen_args)

batch_size = 8

image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
valid_img_generator = image_data_generator.flow(X_test, seed=seed,
                                                batch_size=batch_size)  # Default batch size 32, if not specified here

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
valid_mask_generator = mask_data_generator.flow(y_test, seed=seed,
                                                batch_size=batch_size)  # Default batch size 32, if not specified here


def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


my_generator = my_image_mask_generator(image_generator, mask_generator)

validation_datagen = my_image_mask_generator(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()
for i in range(0, 1):
    image = x[i]
    mask = y[i]
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0])
    plt.show()

steps_per_epoch = 3 * (len(X_train)) // batch_size

history = model.fit_generator(my_generator, validation_data=validation_datagen,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=steps_per_epoch, epochs=25)

model.save("weightsFile.ckpt")
