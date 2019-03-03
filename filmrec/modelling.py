from keras.models import Model, load_model
from keras.layers import Input, Embedding, Reshape
from keras.layers.merge import Dot
import numpy as np
import random
import os

MODEL_FILENAME = "data/trained_embedding_model.h5"


def movie_embedding_model(shapes, embedding_size=50):
    """Form the embedding model

    Kwargs:
        shapes (namedtuple): what shape the dataset is
        embedding_size (int): How big the embedding layers should be

    Returns: keras model

    """
    link = Input(name="link", shape=(1,))
    movie = Input(name="movie", shape=(1,))

    link_embedding = Embedding(
        name="link_embedding", input_dim=shapes.n_links, output_dim=embedding_size
    )(link)

    movie_embedding = Embedding(
        name="movie_embedding", input_dim=shapes.n_movies, output_dim=embedding_size
    )(movie)

    dot = Dot(name="dot_product", normalize=True, axes=2)(
        [link_embedding, movie_embedding]
    )

    merged = Reshape((1,))(dot)

    model = Model(inputs=[link, movie], outputs=[merged])

    model.compile(optimizer="nadam", loss="mse")

    return model


def batchifier(lookups, shapes, positive_samples_per_batch, negative_ratio):
    """Forms batches of data to feed into the model.

    Args:
        lookups (namedtuple): Contains the links
        shapes (namedtuple): the shapes of the dataset
        positive_samples_per_batch (int): Number of positive examples to present
        negative_ratio (int): How many negatives to present

    Returns: generator yielding batches of data

    """
    batch_size = positive_samples_per_batch * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    while True:
        for idx, (link_id, movie_id) in enumerate(
            random.sample(lookups.pairs, positive_samples_per_batch)
        ):
            # For positive samples, we gives these a score of 1
            batch[idx, :] = (link_id, movie_id, 1)

        idx = positive_samples_per_batch

        while idx < batch_size:
            movie_id = random.randrange(shapes.n_movies)
            link_id = random.randrange(shapes.n_links)

            # This is a link that doesn't exist
            if not (link_id, movie_id) in lookups.pairs:
                batch[idx, :] = (link_id, movie_id, -1)
                idx += 1

        np.random.shuffle(batch)

        yield {"link": batch[:, 0], "movie": batch[:, 1]}, batch[:, 2]


def should_train(overwrite=False):
    """Determine if we should train the model or use the one on disk

    Args:
        overwrite (bool): if we want to overwrite the trained model.

    Returns: bool if we should train the model or not
    """
    if overwrite:
        return True
    else:
        if os.path.exists(MODEL_FILENAME):
            return False
        else:
            return True


def train(model, lookups, shapes, positive_samples_per_batch, negative_ratio, epochs):
    """Train the model, saving to disk after training.

    Args:
        model (keras model): the compiled model, ready to train
        lookups (namedtuple): contains the pairs dataset
        shapes (namedtuple): dimension of the dataset
        positive_samples_per_batch (int): number of positive examples in each batch
        negative_ratio (int): how many negative examples to show for each positive
        epochs (int): number of epochs to train for

    Returns: trained model

    """
    model.fit_generator(
        batchifier(
            lookups,
            shapes,
            positive_samples_per_batch=positive_samples_per_batch,
            negative_ratio=negative_ratio,
        ),
        epochs=epochs,
        steps_per_epoch=len(lookups.pairs) // positive_samples_per_batch,
        verbose=2,
    )

    model.save(MODEL_FILENAME)

    return model


def load_trained_model():
    """Loads the model from disk
    Returns: model read from disk

    """
    return load_model(MODEL_FILENAME)
