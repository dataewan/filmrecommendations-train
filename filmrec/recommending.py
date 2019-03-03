def get_popular(lookups):
    """Get a measure of how popular each film is.

    Args:
        lookups (namedtuple): dataset, containing the number of times for each
            link, and also the names of each film

    Returns: dict containing the popularity of each film

    """
    return {
        k: lookups.link_counts[k]
        for k in lookups.movie_to_idx
    }
