# MAEsimplex
(Results from other dataset is coming and update is still continuing)
Anaomaly Detection:
```
python3 train.py --lr=3e-4 --nz=128 --version='planesimplex' --weights=0.5 --cuda=0  --batchsize=256 \
--abnormal_class='plane' --interpolate_points=3 --num_epochs=100
```
Use either batchsize256+lr=3e-4 or batchsize128 and lr=2e-4

abnormal class represent the class of CIFAR10 that is set as anomlay class

interpolate=3 represents 3-points simplex

interpolate=5 represents no simplex loss

interpolate=2 represents 2-points simplex

interpolate=1 represents Berthelot simplex


# Requirment
Python3.7 Most recent pytorch version
