from filmrec import dataprocessing, modelling, recommending

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

    else:
        model = modelling.load_trained_model()

    recommending.write_out_prediction_objects(model, lookups)

    recommending.upload_to_gcs()
