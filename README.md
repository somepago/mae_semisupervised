# Semisupervised - MAEsimplex

Anomaly Detection:
```
python train.py --lr=3e-4 --nz=128 --version='train_no_anom' --weights=0.5 --cuda=0  --batchsize=256 --interpolate_points=3 --num_epochs=100 --update_ratio=2 --anom_pc=0 --abnormal_classes= "['truck']"
```
Use either batchsize256+lr=3e-4 or batchsize128 and lr=2e-4

abnormal class represent the class of CIFAR10 that is set as anomlay class

### interpolation
interpolate=1 represents Berthelot simplex
interpolate=2 represents 2-points simplex
interpolate=3 represents 3-points simplex
interpolate=5 represents no simplex loss

### While training the model, we can use anomalies in the training or not.

--version represents whether training has anomalies or not, you can choose one of these 2 choices for training - ['train_anom', 'train_no_anom']

--anom_pc represents the percentage of anomalous classes to be used in training. Use 0 if you dont want to use anomalies in training

--abnormal_classes is where you choose the anomalies. Following are the different CIFAR classes one can choose as anomalies: ['airplane','automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

### Requirment
Python3.7 Most recent pytorch version
