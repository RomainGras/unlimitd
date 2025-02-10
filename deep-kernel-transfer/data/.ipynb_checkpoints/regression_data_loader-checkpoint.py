import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np



"""
QMUL
"""
    
train_people = ['DennisPNoGlassesGrey','JohnGrey','SimonBGrey','SeanGGrey','DanJGrey','AdamBGrey','JackGrey','RichardHGrey','YongminYGrey','TomKGrey','PaulVGrey','DennisPGrey','CarlaBGrey','JamieSGrey','KateSGrey','DerekCGrey','KatherineWGrey','ColinPGrey','SueWGrey','GrahamWGrey','KrystynaNGrey','SeanGNoGlassesGrey','KeithCGrey','HeatherLGrey']
test_people  = ['RichardBGrey','TasosHGrey','SarahLGrey','AndreeaVGrey','YogeshRGrey']

def num_to_str(num):
    str_ = ''
    if num == 0:
        str_ = '000'
    elif num < 100:
        str_ = '0' + str(int(num))
    else:
        str_ = str(int(num))
    return str_

def get_person_at_curve(person, curve, prefix='filelists/QMUL/images/'):
    faces   = []
    targets = []

    train_transforms = transforms.Compose([transforms.ToTensor()])
    for pitch, angle in curve:
        fname  = prefix + person + '/' + person[:-4] + '_' + num_to_str(pitch) + '_' + num_to_str(angle) +'.jpg'
        img    = Image.open(fname).convert('RGB')
        img    = train_transforms(img)

        faces.append(img)
        pitch_norm = 2 * ((pitch - 60) /  (120 - 60)) -1
        angle_norm = 2 * ((angle - 0)  / (180 - 0)) -1
        targets.append(torch.Tensor([pitch_norm]))

    faces   = torch.stack(faces)
    targets = torch.stack(targets).squeeze()
    return faces, targets

def get_batch_qmul(train_people=train_people, num_samples=19):
    ## generate trajectory
    amp   = np.random.uniform(-3, 3)
    phase = np.random.uniform(-5, 5)
    wave  = [(amp * np.sin(phase + x)) for x in range(num_samples)]
    ## map trajectory to angles/pitches
    angles  = list(range(num_samples))
    angles  = [x * 10 for x in angles]
    pitches = [int(round(((y+3)*10 )+60,-1)) for y in wave]
    curve   = [(p,a) for p, a in zip(pitches, angles)]

    inputs  = []
    targets = []
    for person in train_people:
        inps, targs = get_person_at_curve(person, curve)
        inputs.append(inps)
        targets.append(targs)

    return torch.stack(inputs), torch.stack(targets)
    

"""
MARS datasets
"""

# This is taken from MARS implementation
# TODO temporary remplacement
BASE_DIR = '/home/gridsan/rgras/unlimitd/deep-kernel-transfer' # os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'filelists')

BERKELEY_SENSOR_URL = 'https://www.dropbox.com/sh/y6egx20lod1gsrs/AACyXAk9Ua7SI-q1tpEb1SHba?dl=1'
BERKELEY_SENSOR_DIR = os.path.join(DATA_DIR, 'sensor_data')

ARGUS_CONTROL_URL = 'https://www.dropbox.com/sh/kdzqcw2b0rm34or/AAD2XFzgB2PSjGbNtfNER75Ba?dl=1'
ARGUS_CONTROL_DIR = os.path.join(DATA_DIR, 'argus_data')

class MetaDataset:
    """
    To initiate a random_state, procedure in common for all datasets
    """
    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError

    def generate_meta_test_data(self, n_tasks: int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError
        
        
""" Berkeley Sensor data """


class BerkeleySensorMetaDataset(MetaDataset):

    def __init__(self, random_state=None, separate_train_test_days=True, berkeley_dir=None):
        super().__init__(random_state)
        task_ids = np.arange(46)
        self.random_state.shuffle(task_ids)
        self.train_task_ids = task_ids[:36]
        self.test_task_ids = task_ids[36:]
        self.separate_train_test_days = separate_train_test_days  # whether to also seperate the meta-train and meta-test set by days
        self.data_path = berkeley_dir
        if berkeley_dir is None:
            if not os.path.isdir(BERKELEY_SENSOR_DIR):
                print("Berkeley-Sensor data does not exist in %s" % BERKELEY_SENSOR_DIR)
                download_and_unzip_data(BERKELEY_SENSOR_URL, BERKELEY_SENSOR_DIR)

    def generate_meta_test_data(self, n_tasks=10, n_samples_context=144, n_samples_test=-1):
        task_tuples = self._load_data()

        if n_samples_test == -1:
            n_samples_test = min(2 * self.n_points_per_day, 3 * self.n_points_per_day - n_samples_context)
        else:
            assert n_samples_context + n_samples_test <= 3 * self.n_points_per_day

        test_tuples = []
        for task_id in self.test_task_ids[:n_tasks]:
            x, y = task_tuples[task_id]
            start_idx = -1 * (n_samples_test + n_samples_context)
            x_context, y_context = x[start_idx:-n_samples_test], y[start_idx:-n_samples_test]
            x_test, y_test = x[-n_samples_test:], y[-n_samples_test:]
            test_tuples.append((x_context, y_context, x_test, y_test))
        return test_tuples

    def generate_meta_train_data(self, n_tasks=36, n_samples=-1):
        task_tuples = self._load_data()
        if self.separate_train_test_days:
            if n_samples == -1:
                n_samples = 2 * self.n_points_per_day
            else:
                assert n_samples <= 2 * self.n_points_per_day
        train_tuples = []
        for task_id in self.train_task_ids[:n_tasks]:
            x, y = task_tuples[task_id]
            indices = np.sort(np.random.randint(0, n_samples, 30))
            train_tuples.append((x[indices], y[indices]))
            # train_tuples.append((x[:n_samples], y[:n_samples]))
        return train_tuples

    def _load_data(self, lags=10):
        from scipy.io import loadmat

        if self.data_path is not None:
            data_path = self.data_path + 'berkeley_data.mat'
        else:
            data_path = os.path.join(BERKELEY_SENSOR_DIR, 'berkeley_data.mat')

        data = loadmat(data_path)['berkeley_data']['data'][0][0]
        # replace outlier
        data[4278, 6] = (data[4278 - 1, 6] + data[4278 + 1, 6]) / 2
        n_points_per_day_raw = int(data.shape[0] / 5)
        daytime = np.concatenate([np.arange(n_points_per_day_raw) / n_points_per_day_raw for _ in range(5)])

        # remove first day since it has a break with the remaining 3 days (i.e. day 1, 5, 6, 7, 8]
        data = data[n_points_per_day_raw:]
        daytime = daytime[n_points_per_day_raw:]

        data_tuples = []
        for i in range(data.shape[-1]):
            time_series = data[:, i]
            y = time_series[lags:]
            x = np.stack([time_series[lag: -lags + lag] for lag in range(lags)] + [daytime[lags:]], axis=-1)
            assert x.shape[0] == y.shape[0] == len(time_series) - lags
            # subsample every 5 minutes
            x = x[::10]
            y = y[::10]

            data_tuples.append((x, y))

        self.n_points_per_day = int(data_tuples[0][0].shape[0] / 4)
        return data_tuples

    
""" Argus Control Dataset"""


class ArgusMetaDataset(MetaDataset):

    def __init__(self, random_state=None, task_of_interest='TV', argus_dir=None):
        super().__init__(random_state)
        task_ids_train = np.arange(20)
        task_ids_test = np.arange(4)
        self.random_state.shuffle(task_ids_train)
        self.random_state.shuffle(task_ids_test)
        self.train_task_ids = task_ids_train
        self.test_task_ids = task_ids_test
        self.data_path = argus_dir
        self.task = task_of_interest

        if argus_dir is not None:
            self.data_dir = argus_dir
        elif ARGUS_CONTROL_DIR is not None:
            self.data_dir = ARGUS_CONTROL_DIR
        else:
            raise ValueError("No data directory provided.")

        if not os.path.isdir(self.data_dir):
            print("Argus-Control data does not exist in %s" % self.data_dir)
            download_and_unzip_data(ARGUS_CONTROL_URL, self.data_dir)

        f = open(self.data_dir + '/meta_data_argus_sim.json')

        data = json.load(f)
        self.train_data = data['meta_train'][self.task]
        self.test_data = data['meta_test'][self.task]

    def generate_meta_test_data(self, n_tasks=4, n_samples_context=100, n_samples_test=100):  # fixme
        test_data = []
        indices = np.arange(500)

        for x_context, y_context, x_test, y_test in self.test_data[:n_tasks]:
            self.random_state.shuffle(indices)
            test_data.append((np.array(x_context)[indices[:n_samples_context]],
                              np.array(y_context)[indices[:n_samples_context]],
                              np.array(x_test)[indices[:n_samples_test]],
                              np.array(y_test)[indices[:n_samples_test]]))
        return test_data

    def generate_meta_train_data(self, n_tasks=20, n_samples=100):
        train_data = []
        indices = np.arange(500)
        for x_context, y_context in self.train_data[:n_tasks]:
            self.random_state.shuffle(indices)
            train_data.append((np.array(x_context)[indices[:n_samples], :],
                               np.array(y_context)[indices[:n_samples]]))
        return train_data
    
    
    
def download_and_unzip_data(url, target_dir):
    from urllib.request import urlopen
    from zipfile import ZipFile
    print('Downloading %s' % url)
    # Create the directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    tempfilepath = os.path.join(DATA_DIR, 'tempfile.zip')
    zipresp = urlopen(url)
    with open(tempfilepath, 'wb') as f:
        f.write(zipresp.read())
    zf = ZipFile(tempfilepath)
    print('Extracting to %s' % target_dir)
    zf.extractall(path=target_dir)
    zf.close()
    os.remove(tempfilepath)


""" Data provider """


def provide_data(dataset, seed=28, n_train_tasks=None, n_samples=None, config=None, data_dir=None):
    import numpy as np

    N_TEST_TASKS = 20
    N_VALID_TASKS = 20
    N_TEST_SAMPLES = 200

    # if specified, overwrite default settings
    if config is not None:
        if config['num_test_valid_tasks'] is not None: N_TEST_TASKS = config['num_test_valid_tasks']
        if config['num_test_valid_tasks'] is not None: N_VALID_TASKS = config['num_test_valid_tasks']
        if config['num_test_valid_samples'] is not None:  N_TEST_SAMPLES = config['num_test_valid_samples']

    # """ Prepare Data """       
        
    elif 'argus' in dataset:
        if len(dataset.split('_')) == 2:
            n_train_tasks = int(dataset.split('_')[-1])
        else:
            n_train_tasks = 20
        n_samples_context = 100
        task = 'TV'
        dataset = ArgusMetaDataset(random_state=np.random.RandomState(seed), task_of_interest=task,
                                   argus_dir=data_dir)
        data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_samples_context)
        data_test_valid = dataset.generate_meta_test_data(n_samples_context=n_samples_context, n_samples_test=-1)

        return data_train, data_test_valid, data_test_valid
        
    elif 'berkeley' in dataset:
        if len(dataset.split('_')) == 2:
            n_train_tasks = int(dataset.split('_')[-1])

        dataset = BerkeleySensorMetaDataset(random_state=np.random.RandomState(seed), berkeley_dir=data_dir)

        assert n_samples is None
        n_train_samples = 2 * 144
        n_samples_context = 30  # 144 # corresponds to first day of measurements
        data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)
        data_test_valid = dataset.generate_meta_test_data(n_samples_context=n_samples_context,
                                                          n_samples_test=-1)
        return data_train, data_test_valid, data_test_valid

    else:
        raise NotImplementedError('Does not recognize dataset flag')

    data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)

    data_test_valid = dataset.generate_meta_test_data(n_tasks=N_TEST_TASKS + N_VALID_TASKS,
                                                      n_samples_context=n_context_samples,
                                                      n_samples_test=N_TEST_SAMPLES)
    data_valid = data_test_valid[N_VALID_TASKS:]
    data_test = data_test_valid[:N_VALID_TASKS]
    
    return data_train, data_valid, data_test



def convert_to_right_format(meta_datasets, train_mode: bool = True):
    new_meta_datasets = []
    for meta_data in meta_datasets:
        support_inputs = []
        support_labels = []
        query_inputs = []
        query_labels = []

        # Gather all support & query arrays
        for data in meta_data:
            if train_mode:
                (sX, sY) = data
            else:
                # Only collect query data if not in train mode
                (sX, sY, qX, qY) = data
                query_inputs.append(qX)
                query_labels.append(qY)
                
            support_inputs.append(sX)
            support_labels.append(sY)


        # Convert Python lists -> NumPy arrays -> torch tensors
        # Cast from float 64 to float 32
        support_inputs = torch.from_numpy(np.stack(support_inputs, axis=0)).to(torch.float32).cuda()
        support_labels = torch.from_numpy(np.stack(support_labels, axis=0)).to(torch.float32).cuda()

        if train_mode:
            # For training, store just (support_inputs, support_labels)
            meta_data = (support_inputs, support_labels)
        else:
            # For valid/test, also stack queries
            # Cast from float 64 to float 32
            query_inputs = torch.from_numpy(np.stack(query_inputs, axis=0)).to(torch.float32).cuda()
            query_labels = torch.from_numpy(np.stack(query_labels, axis=0)).to(torch.float32).cuda()

            meta_data = ((support_inputs, support_labels),
                         (query_inputs, query_labels))
            
        new_meta_datasets.append(meta_data)
        
    return new_meta_datasets
        
            

class data_provider():
    def __init__(self, dataset, n_train_support=None, n_test_support=None, n_samples=None):
        # If n_train_support is None, doesn't split meta train datasets
        # Else, splits meta_train datasets into n_train_support support inputs, and the rest as query inputs
        if 'QMUL' in dataset:
            self.n_samples = 19 if n_samples is None else n_samples
            self.n_test_support = 5 if n_test_support is None else n_test_support
        elif 'berkeley' in dataset or 'argus' in dataset:
            # TODO
            self.n_samples = n_samples  # No way to change n_samples 
            self.n_test_support = n_test_support  # No way to change n_test_support 
            
            self.train_tasks, self.valid_tasks, self.test_tasks = provide_data(dataset)
            [self.train_tasks] = convert_to_right_format([self.train_tasks], train_mode=True)
            [self.valid_tasks, self.test_tasks] = convert_to_right_format([self.valid_tasks, self.test_tasks], train_mode=False)

        else:
            raise ValueError("Dataset not recognized")
            
        self.dataset = dataset
        self.n_train_support = n_train_support # 9 for QMUL MAML

    
    def get_train_batch(self):
        if 'QMUL' in self.dataset:
            inputs, targets = get_batch_qmul(train_people=train_people, num_samples=self.n_samples)
            if self.n_train_support is None:
                return inputs.cuda(), targets.cuda()
            # else:
            #    support_ind = list(np.random.choice(list(range(19)), replace=False, size=self.n_train_support))
            #    query_ind   = [i for i in range(19) if i not in support_ind]

                # x_all = inputs.cuda()
                # y_all = targets.cuda()

                # x_support = inputs[:,support_ind,:,:,:].cuda()
                # y_support = targets[:,support_ind].cuda()
                # x_query   = inputs[:,query_ind,:,:,:]
                # y_query   = targets[:,query_ind].cuda()
                # return ((x_support, y_support), (x_query, y_query))
            
        elif 'berkeley' in self.dataset:
            inputs, targets = self.train_tasks
            if self.n_train_support is None:
                return inputs, targets
            
        elif 'argus' in self.dataset:
            inputs, targets = self.train_tasks
            
            permutation = torch.randperm(inputs.size(1))
            # Take the first n of them
            selected_indices = permutation[:20]

            # Subsample along the first dimension using advanced indexing
            #inputs = inputs[:, selected_indices]
            #targets = targets[:, selected_indices]
            
            if self.n_train_support is None:
                return inputs, targets
            # else:
                # TODO : Do better split, these tasks may be splited in a specific way for their implementation in MAML
                # support_ind = list(np.random.choice(list(range(19)), replace=False, size=self.n_train_support))
                # query_ind   = [i for i in range(19) if i not in support_ind]

                # x_all = inputs.cuda()
                # y_all = targets.cuda()

                # x_support = inputs[:,support_ind,:,:,:].cuda()
                # y_support = targets[:,support_ind].cuda()
                # x_query   = inputs[:,query_ind,:,:,:]
                # y_query   = targets[:,query_ind].cuda()
                # return ((x_support, y_support), (x_query, y_query))
        else:
            raise ValueError("Dataset not recognized")
    
    def get_test_batch(self):
        if 'QMUL' in self.dataset:
            inputs, targets = get_batch_qmul(train_people=test_people, num_samples=self.n_samples)
            support_ind = list(np.random.choice(list(range(19)), replace=False, size=self.n_test_support))
            query_ind   = [i for i in range(19) if i not in support_ind]

            x_all = inputs.cuda()
            y_all = targets.cuda()

            x_support = inputs[:,support_ind,:,:,:].cuda()
            y_support = targets[:,support_ind].cuda()
            x_query   = inputs[:,query_ind,:,:,:].cuda()
            y_query   = targets[:,query_ind].cuda()
            return ((x_support, y_support), (x_query, y_query))
        elif 'berkeley' in self.dataset or 'argus' in self.dataset:
            return self.test_tasks
        else:
            raise ValueError("Dataset not recognized")