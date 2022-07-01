# MachineLearningILU
## Description
My thesis project
## Quick start
### Create features vectors
`make_dataset.py` creates the file `data/dataset.csv` with features for matrices from `data/matrices.csv`.

`generate_train_set.py` normalized data from `data/dataset.csv` and split it in two files: `data/train_set.csv` for training and  `data/test_set.csv` for evaluation.

`evaluate_features.py` compared features for two files with dataset.

### Train the neural network
`nn_training/train.py` train neural network on dataset `data/train_set.csv`. Architecture of neural net is in `nn_training/net.py`. Trained model will be saved in the directory `models/`

### Evaluate the neural network
`nn_training/eval.py` evaluate the model saved in `models/model_net.pt` with the `data/test_set.csv`. 
