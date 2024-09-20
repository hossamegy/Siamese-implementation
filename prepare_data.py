import os
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def copy_to_output_dataset(input_path, output_path):
    for person_folder in os.listdir(input_path):
        person_folder_path = os.path.join(input_path, person_folder)
        if os.path.isdir(person_folder_path):
            output_person_folder = os.path.join(output_path, person_folder)
            if not os.path.exists(output_person_folder):
                os.makedirs(output_person_folder)

            for image_file in os.listdir(person_folder_path):
                if image_file.endswith('.jpg'):
                    src_image_path = os.path.join(person_folder_path, image_file)
                    dst_image_path = os.path.join(output_person_folder, image_file)
                    if os.path.exists(dst_image_path):
                        base, ext = os.path.splitext(dst_image_path)
                        dst_image_path = f"{base}_1{ext}"
                    shutil.copy(src_image_path, dst_image_path)


def triplets(folder_paths, max_triplets=7):
    anchor_images = []
    positive_images = []
    negative_images = []

    for person_folder in folder_paths:
        images = [os.path.join(person_folder, img)
                  for img in os.listdir(person_folder)]
        num_images = len(images)

        if num_images < 2:
            continue

        random.shuffle(images)

        for _ in range(max(num_images-1, max_triplets)):
            anchor_image = random.choice(images)

            positive_image = random.choice([x for x in images
                                            if x != anchor_image])

            negative_folder = random.choice([x for x in folder_paths
                                             if x != person_folder])

            negative_image = random.choice([os.path.join(negative_folder, img)
                                            for img in os.listdir(negative_folder)])

            anchor_images.append(anchor_image)
            positive_images.append(positive_image)
            negative_images.append(negative_image)

    return anchor_images, positive_images, negative_images

def split_triplets(anchors,
                   positives,
                   negatives,
                   validation_split=0.2):

    triplets = list(zip(anchors, positives, negatives))

    train_triplets, val_triplets = train_test_split(triplets,
                                                    test_size=validation_split,
                                                    random_state=42)

    return train_triplets, val_triplets


def load_and_preprocess_image(image_path, expand_dims=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    if expand_dims:
        image = np.expand_dims(image, axis=0)
    return image


def batch_generator(triplets, batch_size=32, augment=True):
    total_triplets = len(triplets)
    random_indices = list(range(total_triplets))
    random.shuffle(random_indices)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.2
    )

    for i in range(0, total_triplets, batch_size):
        batch_indices = random_indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            continue 

        batch_triplets = [triplets[j] for j in batch_indices]
        anchor_batch, positive_batch, negative_batch = [], [], []

        for triplet in batch_triplets:
            anchor, positive, negative = triplet

            anchor_image = load_and_preprocess_image(anchor)
            positive_image = load_and_preprocess_image(positive)
            negative_image = load_and_preprocess_image(negative)

            if augment:
                anchor_image = datagen.random_transform(anchor_image)
                positive_image = datagen.random_transform(positive_image)
                negative_image = datagen.random_transform(negative_image)

            anchor_batch.append(anchor_image)
            positive_batch.append(positive_image)
            negative_batch.append(negative_image)

        yield (np.array(anchor_batch), np.array(positive_batch), np.array(negative_batch)), np.zeros((len(batch_indices), 1))