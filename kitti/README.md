# Instructions to train MOVES using CARLA dataset

1) Download the prepared KITTI train and test datast from the link [here](https://www.kaggle.com/datasets/prashk1312/kitti-sequence-extracted-train-test). Copy the data in a folder with name 'kitti' in the current directory. 

2) Go inside the CARLA folder. to train the model run the following command

```
    python MOVES_MMD.py --data [location to the before the 'lidar' folder] --log [logName] --ae_weight [location of the CARLA weight to load the KITTI model]
```
    The weights are stored in the runs/test[logName] folder


3) To evaluate the model on the test set, we need to generate the output LiDAR scan and then evaluate LQI on the ouput. To infer the output and extract LQI for the output run the below commands

``` shell
    python my_eval_save_npy.py --data kitti --ae_weight runs/test[logName]/gen_[index].pth
``` 
This saves the output infered npy in the current directory
Move the npy in a folder named 'npy'

Now generate LQI on the infered npy run the following command
```shell
    cd lqi-files
    python test.py --model kitti.pth --path ../npy/      
```

