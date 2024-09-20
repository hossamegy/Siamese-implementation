from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Flatten, Dense
from tensorflow.keras.applications import EfficientNetV2B3
import tensorflow as tf

# Embedding Model
class EmbeddingModel(Model):
    def __init__(self, embedding_size=128, dropout_rate=0.5):
        super(EmbeddingModel, self).__init__()

        # Use EfficientNetV2B3 as the backbone for feature extraction
        self.feature_extractor = EfficientNetV2B3(input_shape=(128, 128, 3),
                                                  include_top=False,
                                                  weights='imagenet')
        
        # Freeze the layers of EfficientNetV2B3
        self.feature_extractor.trainable = False

        # Add flattening layer for converting 2D feature maps to 1D vectors
        self.flatten = Flatten(name="flatten_layer")

        self.fc1 = Dense(512, activation='relu', name="fc1_dense")
        self.batchnorm1 = BatchNormalization(name="batchnorm1")
        self.dropout1 = Dropout(dropout_rate, name="dropout1")

        self.fc2 = Dense(256, activation='relu', name="fc2_dense")
        self.batchnorm2 = BatchNormalization(name="batchnorm2")
        self.dropout2 = Dropout(dropout_rate, name="dropout2")

        # Output embedding layer with custom embedding size
        self.embedding_output = Dense(embedding_size, name="embedding_output")

    def call(self, inputs):
        # Pass inputs through feature extractor (EfficientNetV2B3)
        x = self.feature_extractor(inputs)
        
        # Flatten the output feature map
        x = self.flatten(x)
        
        # Fully connected layers with batch normalization and dropout
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        # Return the final embedding
        return self.embedding_output(x)

# Custom Triplet Loss Layer
class TripletLayer(Layer):
    def __init__(self, margin=1.0, **kwargs):
        super(TripletLayer, self).__init__(**kwargs)
        self.margin = margin

    def call(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        # Calculate squared Euclidean distance between anchor and positive/negative samples
        positive_distance = tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=1)
        negative_distance = tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=1)

        # Triplet loss: max(positive_distance - negative_distance + margin, 0)
        triplet_loss = tf.maximum(positive_distance - negative_distance + self.margin, 0.0)

        return tf.reduce_mean(triplet_loss)

# Triplet Siamese Model
class TripletSiameseModel(Model):
    def __init__(self, embedding_size=128, margin=1.0, dropout_rate=0.5):
        super(TripletSiameseModel, self).__init__()
        
        # Base model for generating embeddings
        self.embedding_model = EmbeddingModel(embedding_size=embedding_size, dropout_rate=dropout_rate)
        
        # Triplet loss layer with the specified margin
        self.triplet_loss_layer = TripletLayer(margin=margin, name="triplet_loss_layer")

    def call(self, inputs):
        # Unpack the inputs into anchor, positive, and negative examples
        anchor_input, positive_input, negative_input = inputs

        # Generate embeddings for anchor, positive, and negative examples
        anchor_embeddings = self.embedding_model(anchor_input)
        positive_embeddings = self.embedding_model(positive_input)
        negative_embeddings = self.embedding_model(negative_input)

        # Return the triplet loss
        return self.triplet_loss_layer(anchor_embeddings, positive_embeddings, negative_embeddings)

    # Build the model graph for easy visualization
    def build_graph(self):
        anchor_input = Input(shape=(128, 128, 3), name="anchor_input")
        positive_input = Input(shape=(128, 128, 3), name="positive_input")
        negative_input = Input(shape=(128, 128, 3), name="negative_input")
        
        # Create the model
        return Model(inputs=[anchor_input, positive_input, negative_input], 
                     outputs=self.call([anchor_input, positive_input, negative_input]))

# Identity Loss
def identity_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
