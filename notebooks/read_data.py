
import datajoint as dj
dj.config['database.host'] = 'archive.datajoint.io'
dj.config['database.user'] = 'nips'
dj.config['database.password'] = 'nips-submission'

import numpy as np
import seaborn as sns
import sys
sys.path.insert(1, 'C:\\Users\\petko\\Documents\\GitHub\\Sinz2018_NIPS')
sys.path.insert(1, 'C:\\Users\\petko\\Documents\\GitHub\\Sinz2018_NIPS\\attorch')

import matplotlib.pyplot as pltpip 
from collections import namedtuple
from nips2018.movie.parameters import DataConfig, Seed
from nips2018.movie import data
from nips2018.movie.models import Encoder
from tqdm import tqdm
from attorch.train import early_stopping, cycle_datasets

from attorch.dataset import to_variable
#from torch.utils.data import DataLoader
from pprint import pprint
from itertools import chain, repeat, count
from collections import OrderedDict

data.MovieMultiDataset()
DataConfig.AreaLayerClipRawInputResponse()
key = dict(data_hash='5253599d3dceed531841271d6eeba9c5',
               group_id=22,
               seed=2606,
    )

batch_size= 5
val_subsample = None #1000
n_subsample=None



trainsets, trainloaders = DataConfig().load_data(key, tier='train', batch_size=batch_size)
n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])
print('Neurons:',n_neurons)

valsets, valloaders = DataConfig().load_data(key, tier='validation', batch_size=1, key_order=trainsets)

testsets, testloaders = DataConfig().load_data(key, tier='test', batch_size=2, key_order=trainsets)
img_shape = list(trainloaders.values())[0].dataset.img_shape


for readout_key, testloader in testloaders.items():
    print(readout_key)
    for ind, (x_val, beh_val, eye_val, y_val) in enumerate(testloader):
        x = x_val.numpy()
        beh = beh_val.numpy()
        eye = beh_val.numpy()
        print(eye[0,0,:])
        y = beh_val.numpy()
        print(f'{ind}: x:{np.shape(x)}, beh:{np.shape(beh)}, eye:{np.shape(eye)}, y:{np.shape(y)}')
        break

print(testsets)
for readout_key, testset in testsets:
    print(testset)
    for ind,x in enumerate(testset):
        print(x)
print('Neurons:',n_neurons)
quit()
