# Tensorflow serving
# A SavedModel contains a complete TensorFlow program, including weights and computation.

# TF 2.0 saved_model
# https://www.tensorflow.org/alpha/guide/saved_model
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model/save



# cd /Users/j/Dropbox/Learn/kaggle_quora/src

saved_model_cli show --dir sincere/1 --tag_set serve --signature_def serving_default
