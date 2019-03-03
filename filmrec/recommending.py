import numpy as np
from sklearn.neighbors import NearestNeighbors


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
    return {idx: name for name, idx in lookups.movie_to_idx}
