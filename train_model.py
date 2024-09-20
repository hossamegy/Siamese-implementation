import os
import shutil
from tensorflow.keras.optimizers import Adam
from prepare_data import copy_to_output_dataset, triplets, split_triplets, load_and_preprocess_image,  batch_generator
from build_siamese_model import TripletSiameseModel, identity_loss
from trainer import Trainer


FACE_DATA_PATH='/content/Face Data/Face Dataset'
EXTRACTED_FACES_PATH ='/content/Extracted Faces/Extracted Faces'

DATASET = 'images/output_dataset'
if os.path.exists(DATASET):
    shutil.rmtree(DATASET)
os.makedirs(DATASET)


copy_to_output_dataset(FACE_DATA_PATH, DATASET)
copy_to_output_dataset(EXTRACTED_FACES_PATH, DATASET)

person_folders = [os.path.join(DATASET, folder_name)
                  for folder_name in os.listdir(DATASET)]

anchors, positives, negatives = triplets(person_folders)

train_triplets, val_triplets = split_triplets(anchors,
                                              positives,
                                              negatives)
len(train_triplets), len(val_triplets)


loss_tracker = identity_loss
optimizer = Adam()

siamese_net = TripletSiameseModel(embedding_size=256, margin=0.5)

trainer = Trainer(
    batch_generator,
    siamese_net,
    identity_loss,
    optimizer
  )

his_losses = trainer(
    train_triplets,
    val_triplets,
    epochs=30,
    batch_size=64
  )