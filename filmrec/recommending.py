import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from google.cloud import storage
import os

PICKLE_FILENAME = "data/pickled_model.pkl"

def get_popular(lookups):
    """Get a measure of how popular each film is.

    Args:
        lookups (namedtuple): dataset, containing the number of times for each
            link, and also the names of each film

    Returns: dict containing the popularity of each film

    """
    return {k: lookups.link_counts[k] for k in lookups.movie_to_idx}


def get_neighbours(lookups, model):
    """Get the nearest neighbours for each film.

    Args:
        lookups (namedtuple): Lookups against the dataset
        model (keras model): Trained keras model

    Returns: the nearest neighbour IDs for each film

    """
    movie_layer = model.get_layer("movie_embedding")
    movie_weights = movie_layer.get_weights()[0]
    movie_lengths = np.linalg.norm(movie_weights, axis=1)

    normalised_movies = (movie_weights.T / movie_lengths).T

    neighbours_model = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(
        normalised_movies
    )

    # calculate the distances and N nearest neighbours for each of the
    # normalised movies
    distances, neighbours = neighbours_model.kneighbors(normalised_movies)

    lookup = {idx: nneighbours for idx, nneighbours in enumerate(neighbours[1:, :])}
    return lookup


def get_reverse_movie_lookup(lookups):
    """Instead of having movie->id, get one that goes the other way
    Args:
        lookups (namedtuple): contains the original lookup
    
    Returns: reversed lookup

    """
    return {idx: name for name, idx in lookups.movie_to_idx.items()}


def write_out_prediction_objects(model, lookups):
    """Write out the pickle object that contains everything we need to serve
        lookups

    Args:
        model (keras model): trained model with the embedding weights
        lookups (namedtuple): lookup objects

    Returns: dict with the movie popularity, the neighbours and lookup to get
        the title. This has also been written to disk.

    """
    neighbours = get_neighbours(lookups, model)
    id_to_movie = get_reverse_movie_lookup(lookups)
    popular_films = get_popular(lookups)

    output = {
        "neighbours": neighbours,
        "id_to_movie": id_to_movie,
        "popular_films": popular_films,
    }

    with open(PICKLE_FILENAME, "wb") as f:
        pickle.dump(output, f)

    return output


def upload_to_gcs():
    """Uploads the pickled file to gcs
    """
    client = storage.Client(project="filmreccommendations")
    bucket = client.get_bucket("filmreccommendations.appspot.com")
    blob = bucket.blob(os.path.basename(PICKLE_FILENAME))
    blob.upload_from_filename(PICKLE_FILENAME)
