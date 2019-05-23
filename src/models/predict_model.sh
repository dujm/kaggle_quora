# tf serving using docker
# 0 Use saved_model in /Users/j/Dropbox/Learn/kaggle_quora/src/models/sincere

# 1. Start docker
# 2. Run docker image with the model
docker run -t --rm -p 8501:8501 \
    -v "/Users/j/Dropbox/Learn/kaggle_quora/src/models/sincere:/models/sincere" \
    -e MODEL_NAME=sincere \
    tensorflow/serving &

# 3. Check http://localhost:8501/v1/models/sincere, confirms working
'''
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}

'''
