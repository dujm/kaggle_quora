# 1. clone tfserving repo
# cd  localpath
git clone https://github.com/tensorflow/serving.git

# 2. Build tfserving locally
cd serving
tools/run_in_docker.sh bazel build -c opt tensorflow_serving/...

# 3. Binaries are placed in the bazel-bin directory, and can be run using a command like
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server

# 4. To test your build, execute
tools/run_in_docker.sh bazel test -c opt tensorflow_serving/...
