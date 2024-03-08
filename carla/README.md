Instructions to train MOVES using CARLA dataset

1) Download the paired correspondence CARLA training datast from the line [here](https://www.kaggle.com/datasets/ssahoodotinfinity/carla-64-training) (9 GB) here and the test dataset from [here](https://www.kaggle.com/datasets/ssahoodotinfinity/carla64-preprocessed-range-image-dataset) (4 GB). Copy the training data in a folder with name 'train' in the current directory. For the test, put the test data in a folder 'test'

2) Go inside the CARLA folder. to train the model run the following command

    python MOVES.py --data [location to the before the train folder] --log [name of the log]

    The weights are stored in the runs/test folder


3) To evaluate the model on the test set, run the following command

    python eval_chamfer.py --ae_weight runs/testtest/gen_[index].pth --data [path to the before the test folder]
