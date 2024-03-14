# Instructions to train MOVES using CARLA dataset

1) Download the ARD training and testing dataset from [here](https://www.kaggle.com/datasets/prashk1312/ati-preprocess). Copy the datasetin a folder with name 'lidar' in the current directory. 

2) Go inside the CARLA folder. to train the model run the following command

```shell
    python MOVES.py --data [location to the before the 'lidar' folder] --log [name of the log]
```
    The weights are stored in the runs/test folder


3) To evaluate the model on the test set, run the following command

```shell
    python eval_chamfer_and_save_npy.py --ae_weight runs/test[logfolder]/gen_[index].pth --data [path to the before the test folder]
```
