
## Aim:
Use Tensorflow Serving to serve a TF2.0 NLP model  

------
## Table of Contents
- [Usage](#usage)
- [Files](#files)
- [Code style](#code-style)
- [Prerequisites](#prerequisites)
- [Reference](#reference)
- [License](#license)


------
## Usage
1. Clone the repo

```
git clone git@github.com:dujm/kaggle_quora.git

# Remove my git directory
cd kaggle_quora
rm -r .git/
```

2. Install packages

```
pip install -r requirements.txt
```

3. Download the Kaggle Quora dataset
 * [Download from Kaggle Website](https://www.kaggle.com/c/quora-insincere-questions-classification/data)

 * [Or install Kaggle API](https://dujm.github.io/datasciences/kaggle) and run:

    ```
    bash src/data/download-dataset.sh
    unzip src/data/embeddings.zip
    ```

4. Build tensorflow serving locally

```
bash src/models/00build-tfserve.sh
```

5. Train a test NLP model and save the model as a **Tensorflow saved model**

```
python src/models/01train-saved-model.py
```

6. Serve the model locally using Tensorflow Serving

```
bash src/models/02tf-serve-model.sh
```
TBC

------
##  Files

     ├── LICENSE
     ├── README.md
     ├── src
     ├── test_environment.py
     │   ├── data
     │   │   ├── download-dataset.sh
     │   │   ├── embeddings
     │   │   │   ├── embeddings_index.npy
     │   │   │   ├── glove.840B.300d
     │   │   │   │   └── glove.840B.300d.txt
     │   │   │   ├── Other two embedding files (Not used here)
     │   │   ├── input
     │   │   │   ├── test.csv
     │   │   │   └── train.csv
     │   ├── models
     │   │   ├── 00build-tfserve.sh
     │   │   ├── 01train-saved-model.py
     │   │   ├── 02tf-serve-model.sh
     │   │   ├── sincere
     │   │   │   └── 1
     │   │   │       ├── assets
     │   │   │       ├── saved_model.pb
     │   │   │       └── variables
     │   │   │           ├── variables.data-00000-of-00001
     │   │   │           └── variables.index
     │   │   └── utils.py
     └──test_environment.py


------
## Code style
The information about code style in python is documented in this two links [python-developer-guide](https://github.com/oceanprotocol/dev-ocean/blob/master/doc/development/python-developer-guide.md) and [python-style-guide](https://github.com/oceanprotocol/dev-ocean/blob/master/doc/development/python-style-guide.md).

------
## Prerequisites
Python 3

------
## Reference
 * [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
 * [Using the SavedModel format](https://www.tensorflow.org/alpha/guide/saved_model)
 * [tf.saved_model.save](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/saved_model/save)
 * [TensorFlow Serving](https://github.com/tensorflow/serving)

------
## License
The MIT License (MIT)
