import numpy as np
from tqdm import tqdm
import math
import tensorflow as tf

class Trainer:
    def __init__(self, batch_generator, model, loss_fun, optimizer):
        """
        Initializes the Trainer class with a model, loss function, and optimizer.
        Args:
            model: Siamese model.
            loss_fun: The loss function to optimize.
            optimizer: The optimizer to apply during training.
        """
        self.model = model
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.batch_generator = batch_generator

    def __call__(self, train_triplets, val_triplets, epochs, batch_size):
        """
        Executes the training and validation loops for the specified epochs and batch size.
        Args:
            train_triplets: Training data in triplet format (anchor, positive, negative).
            val_triplets: Validation data in triplet format (anchor, positive, negative).
            epochs: Number of epochs to train the model.
            batch_size: Size of each batch during training and validation.
        Returns:
            A tuple containing lists of training and validation losses for each epoch.
        """
        train_epochs_losses = []
        val_epochs_losses = []

        # Compute steps per epoch
        train_steps_per_epoch = math.ceil(len(train_triplets) / batch_size)
        val_steps_per_epoch = math.ceil(len(val_triplets) / batch_size)

        for epoch in range(epochs):
            # Initialize generators for training and validation
            train_generator = self.batch_generator(train_triplets, batch_size=batch_size, augment=True)
            val_generator = self.batch_generator(val_triplets, batch_size=batch_size, augment=False)
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training step
            train_losses = self.train_step_for_one_epoch(train_generator, train_steps_per_epoch)
            avg_train_loss = np.mean(train_losses)
            train_epochs_losses.append(avg_train_loss)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # Validation step
            val_losses = self.perform_validation(val_generator, val_steps_per_epoch)
            avg_val_loss = np.mean(val_losses)
            val_epochs_losses.append(avg_val_loss)
            print(f"Average Validation Loss: {avg_val_loss:.4f}")

        return train_epochs_losses, val_epochs_losses

    @tf.function
    def apply_gradients(self, anchors, positives, negatives, labels):
        """
        Applies gradients and updates the model weights using the optimizer.
        Args:
            anchors: Anchor images (input).
            positives: Positive images (input).
            negatives: Negative images (input).
            labels: Labels for the triplet loss.
        Returns:
            The predicted values and the computed loss for the batch.
        """
        with tf.GradientTape() as tape:
            y_pred = self.model([anchors, positives, negatives], training=True)
            loss = self.loss_fun(labels, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return y_pred, loss

    def train_step_for_one_epoch(self, train_generator, train_steps_per_epoch):
        """
        Performs one epoch of training across all batches.
        Args:
            train_generator: Generator that yields batches of training data.
            train_steps_per_epoch: Number of training steps in one epoch.
        Returns:
            A list of losses for each batch in the epoch.
        """
        losses = []
        pbar = tqdm(total=train_steps_per_epoch, position=0, leave=True)
        for step, ((anchors, positives, negatives), labels) in enumerate(train_generator):
            # Convert to TensorFlow tensors if not already
            anchors, positives, negatives, labels = map(tf.convert_to_tensor, (anchors, positives, negatives, labels))
            
            y_pred, loss = self.apply_gradients(anchors, positives, negatives, labels)
            losses.append(loss)  # Convert Tensor loss to a numpy float

            pbar.set_description(f"Training Loss: {float(loss):.4f}")
            pbar.update(1)
        pbar.close()
        return losses
        
    
    def perform_validation(self, val_generator, val_steps_per_epoch):
        """
        Performs validation across all batches.
        Args:
            val_generator: Generator that yields batches of validation data.
            val_steps_per_epoch: Number of validation steps in one epoch.
        Returns:
            A list of losses for each batch in the validation epoch.
        """
        losses = []
        pbar = tqdm(total=val_steps_per_epoch, position=0, leave=True)
        for step, ((anchors, positives, negatives), labels) in enumerate(val_generator):
            # Convert to TensorFlow tensors if not already
            anchors, positives, negatives, labels = map(tf.convert_to_tensor, (anchors, positives, negatives, labels))

            y_pred = self.model([anchors, positives, negatives], training=False)
            loss = self.loss_fun(labels, y_pred)
            
            # Convert the loss tensor to a numpy float for logging and storing
            loss_value = loss.numpy()  # Ensure the tensor is converted to a NumPy value
            losses.append(loss_value)

            pbar.set_description(f"Validation Loss: {loss_value:.4f}")
            pbar.update(1)
        pbar.close()
        return losses