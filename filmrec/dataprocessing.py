import json
from collections import Counter, namedtuple

Lookups = namedtuple(
    "Lookups", ["pairs", "top_links", "movie_to_idx", "link_to_idx", "link_counts"]
)
Shapes = namedtuple("Shapes", ["n_movies", "n_links"])


def read_data(filename):
    """Read ndjson file containing the movies and links

    Args:
        filename (string): path of the data file

    Returns: list containing the movie data

    """
    with open(filename) as fin:
        movies = [json.loads(l) for l in fin]

    return movies


def get_top_links(movies, N=3):
    """Get the top links out of the dataset, filtering out anything less than N.

    Args:
        movies (list): list containing the movie data
        N (int): cutoff value

    Returns: list containing the top links only, also counter object for how
        often each link happened

    """
    link_counts = Counter()
    for movie in movies:
        link_counts.update(movie[2])

    top_links = [link for link, c in link_counts.items() if c >= N]

    return top_links, link_counts


def get_movie_to_idx(movies):
    """Get a lookup dict converting the titles of the films to ids

    Args:
        movies (list): movie json data

    Returns: dict from title: id

    """
    return {movie[0]: idx for idx, movie in enumerate(movies)}


def get_link_to_idx(top_links):
    """Get a lookup dictionary converting the names of the links to ids

    Args:
        top_links (list): containing the top links only

    Returns: dict from link name: id

    """
    return {link: idx for idx, link in enumerate(top_links)}


def get_pairs(movies, movie_to_idx, link_to_idx):
    """Get all the movie_id: movie_id pairs where there are links from one page
        to the next

    Args:
        movies (list): list containing the movie json data
        movie_to_idx (dict): lookup containing the title: id
        link_to_idx (dict): lookup containing the link name: id

    Returns: set of tuples containing link_id: movie_id

    """
    pairs = []
    for movie in movies:
        pairs.extend(
            (link_to_idx[link], movie_to_idx[movie[0]])
            for link in movie[2]
            if link in link_to_idx
        )

    return set(pairs)


def form_lookups(movies):
    """There are four lookups that I need to form:
    1. pairs -- which links come from each movie page
    2. top_links -- the top links, filtered so that only links that happen a
        few times are here
    3. movie_to_idx -- lookup dict converting the movie title to number
    4. link_to_idx -- lookup dict converting the link name to number
    5. link_counts -- how often each link happened

    Args:
        movies (list): list containing the json movie data

    Returns: pairs, top_links, movie_to_idx, link_to_idx, link_counts

    """
    top_links, link_counts = get_top_links(movies)
    movie_to_idx = get_movie_to_idx(movies)
    link_to_idx = get_link_to_idx(top_links)
    pairs = get_pairs(movies, movie_to_idx, link_to_idx)

    return Lookups(
        pairs=pairs,
        top_links=top_links,
        movie_to_idx=movie_to_idx,
        link_to_idx=link_to_idx,
        link_counts=link_counts,
    )


def get_shapes(lookups):
    """Get the dimensions of the dataset, so that we make a network of the right size
    Args:
        lookups (namedtuple): the lookups from the dataset

    Returns: n_movies, n_links
    
    """
    n_movies = len(lookups.movie_to_idx)
    n_links = len(lookups.top_links)
    return Shapes(n_movies=n_movies, n_links=n_links)
