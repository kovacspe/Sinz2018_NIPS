
import datajoint as dj
dj.config['database.host'] = 'archive.datajoint.io'
dj.config['database.user'] = 'nips'
dj.config['database.password'] = 'nips-submission'

import numpy as np
import seaborn as sns
import sys
sys.path.insert(1, 'C:\\Users\\petko\\Documents\\GitHub\\Sinz2018_NIPS')
sys.path.insert(1, 'C:\\Users\\petko\\Documents\\GitHub\\Sinz2018_NIPS\\attorch')

import matplotlib.pyplot as plt
from nips2018.utils import set_seed
from nips2018.utils.measures import corr
from collections import namedtuple
from nips2018.movie.parameters import DataConfig, Seed
from nips2018.architectures.readouts import SpatialTransformerPooled3dReadout, ST3dSharedGridStopGradientReadout
from nips2018.architectures.cores import StackedFeatureGRUCore, Stacked3dCore
from nips2018.architectures.shifters import StaticAffineShifter
from nips2018.architectures.modulators import GateGRUModulator 
from nips2018.movie import data
from nips2018.movie.models import Encoder
from nips2018.architectures.base import CorePlusReadout3d
import torch
from itertools import count
from tqdm import tqdm
from attorch.layers import elu1, Elu1
from attorch.train import early_stopping, cycle_datasets

from attorch.dataset import to_variable
from torch.utils.data import DataLoader
from pprint import pprint
from itertools import chain, repeat
from attorch.losses import PoissonLoss3d
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict

data.MovieMultiDataset()
DataConfig.AreaLayerClipRawInputResponse()
key = dict(data_hash='5253599d3dceed531841271d6eeba9c5',
               group_id=22,
               seed=2606,
    )

N_GPU = torch.cuda.device_count()
print('NGPU: ',N_GPU)
N_GPU = 1
batch_size= 5
val_subsample = None #1000
n_subsample=None

set_seed(key['seed'])

trainsets, trainloaders = DataConfig().load_data(key, tier='train', batch_size=batch_size)
n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])

valsets, valloaders = DataConfig().load_data(key, tier='validation', batch_size=1, key_order=trainsets)

testsets, testloaders = DataConfig().load_data(key, tier='test', batch_size=2, key_order=trainsets)
img_shape = list(trainloaders.values())[0].dataset.img_shape


for batch_no, (readout_key, *data) in enumerate(cycle_datasets(trainloaders, requires_grad=False)):
    print(f'batch_n={batch_no}, readout={readout_key}, {data}')
quit()
from nips2018.movie import parameters
best = Encoder * (dj.U('group_id').aggr(Encoder, best = 'max(val_corr)')) & 'val_corr >= best and group_id=22'
best * parameters.CoreConfig.StackedFeatureGRU

core = StackedFeatureGRUCore(input_channels=img_shape[1], hidden_channels=12, rec_channels=36,
                    input_kern=7, hidden_kern=3, rec_kern=3, layers=3, 
                    gamma_input=50, gamma_hidden=.1, gamma_rec=.0, momentum=.1,
                             skip=2, bias=False, batch_norm=True, pad_input=True
                   )
ro_in_shape = CorePlusReadout3d.get_readout_in_shape(core, img_shape)

readout = ST3dSharedGridStopGradientReadout(ro_in_shape, 
                                               n_neurons, 
                                               positive=False,  
                                               gamma_features=1., 
                                               pool_steps=2,
                                                kernel_size=4,
                                                stride=4,
                                            gradient_pass_mod=3
                                           )
shifter = StaticAffineShifter(n_neurons, input_channels=2, hidden_channels=2, bias=True, gamma_shifter=0.001)
modulator = GateGRUModulator(n_neurons, gamma_modulator=0.0, hidden_channels=50, offset=1, bias=True)
model = CorePlusReadout3d(core, readout, nonlinearity=Elu1(), 
                        shifter=shifter, modulator=modulator, burn_in=15)
model


img_shape = list(trainloaders.values())[0].dataset.img_shape



criterion = PoissonLoss3d()
n_datasets = len(trainloaders)
acc = 1 # accumulate gradient over this many steps


# --- setup objective
grad_passes = 0
for ro in model.readout.values():
    grad_passes += int(not ro.stop_grad)

def full_objective(model, readout_key, inputs, beh, eye_pos, targets):
    outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
    return (criterion(outputs, targets)
            + (model.core.regularizer() / grad_passes if not model.readout[readout_key].stop_grad else 0)
            + model.readout.regularizer(readout_key).cuda(0)
            + (model.shifter.regularizer(readout_key) if model.shift else 0)
            + (model.modulator.regularizer(readout_key) if model.modulate else 0)) / acc

# --- initialize
stop_closure = Encoder().get_stop_closure(valloaders, subsamp_size=val_subsample)

mu_dict = OrderedDict([
    (k, dl.dataset.mean_trial().responses) for k, dl in trainloaders.items()
])
model.readout.initialize(mu_dict)
model.core.initialize()


if model.shifter is not None:
    biases = OrderedDict([
        (k, -dl.dataset.mean_trial().eye_position) for k, dl in trainloaders.items()
    ])
    model.shifter.initialize(bias=biases)
if model.modulator is not None:
    model.modulator.initialize()

model = model.cuda()

def train(model, objective, optimizer, stop_closure, trainloaders, epoch=0, post_epoch_hook=None,
          interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
          restore_best=True, accumulate_gradient=1):
    assert not isinstance(optimizer, torch.optim.LBFGS), "We don't BFGS at the moment. "
    optimizer.zero_grad()
    iteration = 0
    assert accumulate_gradient > 0, 'accumulate_gradient needs to be > 0'

    for epoch, val_obj in early_stopping(model, stop_closure,
                                         interval=interval, patience=patience,
                                         start=epoch, max_iter=max_iter, maximize=maximize,
                                         tolerance=tolerance, restore_best=restore_best):
        for batch_no, (readout_key, *data) in \
                tqdm(enumerate(cycle_datasets(trainloaders, requires_grad=False, cuda=cuda)),
                     desc='Training  | Epoch {}'.format(epoch)):
            obj = objective(model, readout_key, *data)
            obj.backward()
            if iteration % accumulate_gradient == accumulate_gradient - 1:
                optimizer.step()
                optimizer.zero_grad()
            iteration += 1

        if post_epoch_hook is not None:
            model = post_epoch_hook(model, epoch)
    return model, epoch


    
epoch = 0
# --- train core, modulator, and readout but not shifter
schedule = [0.005, 0.001]

for opt, lr in zip(repeat(torch.optim.Adam), schedule):
    print('Training with learning rate', lr)
    optimizer = opt(model.parameters(), lr=lr)

    model, epoch = train(model, full_objective, optimizer,
                                   stop_closure, trainloaders,
                                   epoch=epoch,
                                   max_iter=100,
                                   interval=4,
                                   patience=4,
                                   accumulate_gradient=acc
                                   )
model.eval()


PerformanceScores = namedtuple('PerformanceScores', ['pearson'])


def compute_scores(y, y_hat, axis=0):
    pearson = corr(y, y_hat, axis=axis)
    return PerformanceScores(pearson=pearson)

def compute_predictions(loader, model, readout_key, reshape=True, stack=True, subsamp_size=None, return_lag=False):
    y, y_hat = [], []
    for x_val, beh_val, eye_val, y_val in tqdm(to_variable(loader, filter=(True, True, True, False),
                                                           cuda=True, volatile=True), desc='predictions'):
        neurons = y_val.size(-1)
        if subsamp_size is None:
            y_mod = model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val).data.cpu().numpy()
        else:
            y_mod = []
            neurons = y_val.size(-1)
            for subs_idx in slice_iter(neurons, subsamp_size):
                y_mod.append(
                    model(x_val, readout_key, eye_pos=eye_val,
                          behavior=beh_val, subs_idx=subs_idx).data.cpu().numpy())
            y_mod = np.concatenate(y_mod, axis=-1)

        lag = y_val.shape[1] - y_mod.shape[1]
        if reshape:
            y.append(y_val[:, lag:, :].numpy().reshape((-1, neurons)))
            y_hat.append(y_mod.reshape((-1, neurons)))
        else:
            y.append(y_val[:, lag:, :].numpy())
            y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat)
    print(y)
    if not return_lag:
        return y, y_hat
    else:
        return y, y_hat, lag
        
def compute_test_scores(testloaders, model, readout_key):
    loader = testloaders[readout_key]

    y, y_hat = compute_predictions(loader, model, readout_key, reshape=True, stack=True, subsamp_size=None)
    return compute_scores(y, y_hat)  # scores is a named tuple



def compute_test_score_tuples(key, testloaders, model):
    scores, unit_scores = [], []
    for readout_key, testloader in testloaders.items():
        perf_scores = compute_test_scores(testloaders, model, readout_key)

        member_key = (data.MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)  # get other fields
        member_key.update(key)
        unit_ids = testloader.dataset.neurons.unit_ids
        member_key['neurons'] = len(unit_ids)
        member_key['pearson'] = perf_scores.pearson.mean()

        scores.append(member_key)
        unit_scores.extend([dict(member_key, unit_id=u, pearson=c) for u, c in zip(unit_ids, perf_scores.pearson)])
    return scores, unit_scores

scores, unit_scores = compute_test_score_tuples(key, testloaders, model)
print(scores)