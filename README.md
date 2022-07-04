# MachineLearningILU
## Description
My thesis project
## Quick start
### Create features vectors
Paths to all files in this part can be changed in `dataset/data_files.py` 

`dataset/make_dataset.py` creates the file `data/dataset.csv` with features for matrices from `data/matrices.csv`.
By setting the argument `ss` it will process matrices from https://sparse.tamu.edu/, with the argument `random` it will process random matrices from matlab project.  

`dataset/generate_train_set.py` normalized data from `data/dataset.csv` and split it in two files: `data/train_set.csv` for training and  `data/test_set.csv` for evaluation.

`dataset/evaluate_features.py` compared features for two files with dataset.

### Machine Learning
`scikit/random_forest.py` train random forest algorithm on dataset `data/train_set.csv`. 

### Train the neural network
`nn_training/train.py` train neural network on dataset `data/train_set.csv`. 
Architecture of neural net is in `nn_training/net.py`. Trained model will be saved in the directory `models/`

### Evaluate the neural network
`nn_training/eval.py` evaluate the model saved in `models/model_net.pt` with the `data/test_set.csv`. 
