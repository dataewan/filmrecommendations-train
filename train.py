from filmrec import dataprocessing, modelling

if __name__ == "__main__":
    positive_samples_per_batch = 512
    movies = dataprocessing.read_data("data/wp_movies_10k.ndjson")
    lookups = dataprocessing.form_lookups(movies)
    shapes = dataprocessing.get_shapes(lookups)

    if modelling.should_train():
        model = modelling.movie_embedding_model(shapes)
        modelling.train(
            model,
            lookups,
            shapes,
            positive_samples_per_batch=512,
            negative_ratio=10,
            epochs=15,
        )
