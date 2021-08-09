# Don't bother reading this!  Just utility functions.
import shutil, os, pathlib, pickle, sys, math, importlib, json.tool, argparse, atexit, builtins
import numpy as np
from glob import glob
from os.path import join, exists, isdir
from tqdm import tqdm
from itertools import product
from datetime import datetime

def flip_bool(bool_in):
  return bool((int(bool_in) + 1) % 2)

def lvmap(f, arr, axis=None):
    if axis is None:
        f = np.vectorize(f)
        return f(arr)
    else:
        return np.apply_along_axis(f,axis=axis,arr=arr)

def init_except_hook(gpu_id=None, filename=None, test=False):
    def my_except_hook(exctype, value, traceback):
        print('\n\n########### ERROR ###########')
        print('Emailing you that an error has occurred...')
        # update_gpu_log(gpu_id, 'open')
        sys.__excepthook__(exctype, value, traceback)
        send_email()
            # t.send(f'ERROR: {os.path.basename(__file__) if filename is None else filename} failed')
    sys.excepthook = my_except_hook

def init_exit_hook(gpu_id=None, test=False):
    def my_exit_hook():
        # update_gpu_log(gpu_id, 'open')
        if not test:
            send_email()
            # t.send('Finished!')
    atexit.register(my_exit_hook)

def send_email(subject='Hi there', text='Hello!', secrets_path='/z/abwilf/dw/mailgun_secrets.json'):
    secrets = load_json(secrets_path)
    return requests.post(
        secrets['url'],
        auth=("api", secrets['api_key']),
        data={"from": secrets['from_addr'],
            "to": secrets['to_addr'],
            "subject": subject,
            "text": text})

class Runtime():
    def __init__(self):
        self.start_time = datetime.now()
    def get(self):
        end_time = datetime.now()
        sec = (end_time - self.start_time).seconds
        days = int(sec/(3600*24))
        hrs = int(sec/3600)
        mins = int((sec % 3600)/60)

        days_str = f'{days} days, ' if days > 0 else ''
        hrs_str = f'{hrs} hrs, ' if hrs > 0 else ''
        # print(f'\nEnd time: {end_time}')
        print(f'Runtime: {days_str}{hrs_str}{mins} mins')

def init_except_hook(gpu_id=None, filename=None, test=False):
    def my_except_hook(exctype, value, traceback):
        print('\n\n########### ERROR ###########')
        print('Emailing you that an error has occurred...')
        update_gpu_log(gpu_id, 'open')
        sys.__excepthook__(exctype, value, traceback)
        if not test:
            send_email()
            # t.send(f'ERROR: {os.path.basename(__file__) if filename is None else filename} failed')
    sys.excepthook = my_except_hook

def init_exit_hook(gpu_id=None, test=False):
    def my_exit_hook():
        update_gpu_log(gpu_id, 'open')
        if not test:
            send_email()
            # t.send('Finished!')
    atexit.register(my_exit_hook)

def send_email(subject='Hi there', text='Hello!', secrets_path='./mailgun_secrets.json'):
    secrets = load_json(secrets_path)
    return requests.post(
        secrets['url'],
        auth=("api", secrets['api_key']),
        data={"from": secrets['from_addr'],
            "to": secrets['to_addr'],
            "subject": subject,
            "text": text})

def ar(a):
    return np.array(a)

def rmtree(dir_path):
    print(f'Removing {dir_path}')
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    # else:
        # print(f'{dir_path} is not a directory, so cannot remove')

def npr(x, decimals=2):
    '''Round'''
    return np.round(x, decimals=decimals)

def nprs(x, decimals=2, scale=100):
    '''Round & scale'''
    return np.round(x*scale, decimals=decimals)

def int_to_str(*keys):
    return [list(map(lambda elt: str(elt), key)) for key in keys]

def rm_mkdirp(dir_path, overwrite, quiet=False):
    if os.path.isdir(dir_path):
        if overwrite:
            if not quiet:
                print('Removing ' + dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)

        else:
            print('Directory ' + dir_path + ' exists and overwrite flag not set to true.  Exiting.')
            exit(1)
    if not quiet:
        print('Creating ' + dir_path)
    pathlib.Path(dir_path).mkdir(parents=True)

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def rmfile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def rglob(dir_path, pattern):
    return list(map(lambda elt: str(elt), pathlib.Path(dir_path).rglob(pattern)))

def move_matching_files(dir_path, pattern, new_dir, overwrite):
    rm_mkdirp(new_dir, True, overwrite)
    for elt in rglob(dir_path, pattern):
        shutil.move(elt, new_dir)

def subset(a, b):
    return np.min([elt in b for elt in a]) > 0

def subsets_eq(a,b):
    return subset(a,b) and subset(b,a)

def dict_at(d):
    k = lkeys(d)[0]
    return k, d[k]

def list_gpus():
    return tf.config.experimental.list_physical_devices('GPU')

def save_pk(file_stub, pk, protocol=4):
    filename = file_stub if '.pk' in file_stub else f'{file_stub}.pk'
    rmfile(filename)
    with open(filename, 'wb') as f:
        pickle.dump(pk, f, protocol=protocol)

def load_pk(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        return load_pk_old(filename)

def load_pk_old(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

def get_dir(path, silent=True):
    if '.' not in path and not silent:
        print(f'NOTE: {path} is not a file, creating dir with just {path}')
    else:
        path = '/'.join(path.split('/')[:-1])
    return path

def get_ints(*keys):
    return [int(key) for key in keys]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)

def load_json(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    with open(filename) as json_file:
        return json.load(json_file)

def lfilter(fn, iterable):
    return list(filter(fn, iterable))

def lkeys(obj):
    return list(obj.keys())

def lvals(obj):
    return list(obj.values())

def lmap(fn, iterable):
    return list(map(fn, iterable))

def arlmap(fn, iterable):
    return ar(list(map(fn, iterable)))

def arlist(x):
    return ar(list(x))

def llmap(fn, iterable):
    return list(map(lambda elt: fn(elt), iterable))

def sort_dict(d, reverse=False):
    return {k: v for k,v in sorted(d.items(), key=lambda elt: elt[1], reverse=reverse)}

def csv_path(sym):
    return join('csvs', f'{sym}.csv')

def is_unique(a):
    return len(np.unique(a)) == len(a)

def lists_equal(a,b):
    return np.all([elt in b for elt in a]) and np.all([elt in a for elt in b])

def split_arr(cond, arr):
    return lfilter(cond, arr), lfilter(lambda elt: not cond(elt), arr)

def lzip(*keys):
    return list(zip(*keys))

def zero_pad_to_length(data, length):
    padAm = length - data.shape[0]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,padAm), (0,0)), 'constant')