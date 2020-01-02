Specify abnormal class for cifar10
Interpolate = 3 means 3points-simplex 
Interpolate = 5 means no simplex
Interpolate = 2 means 2points simplex 
Interpolate = 1 menas Goodfellow's interpolation 

srun python3 train.py --lr=3e-4 --nz=128 --version='birdsimplex' --weights=0.5 --cuda=0  --batchsize=256 \
--abnormal_class='bird' --interpolate_points=3 --num_epochs=100
