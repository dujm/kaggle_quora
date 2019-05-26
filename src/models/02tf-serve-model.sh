# tf serving using docker
# 0 Use saved_model in /Users/j/Dropbox/Learn/kaggle_quora/src/models/sincere
# Run docker image
docker run -t --rm -p 8501:8501 \
    -v "/Users/j/Dropbox/Learn/kaggle_quora/src/models/sincere:/models/sincere" \
    -e MODEL_NAME=sincere \
    tensorflow/serving &

# Confirm model is served 
curl http://localhost:8501/v1/models/sincere
