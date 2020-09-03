import h5py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def read_file(name):
    f = h5py.File('{}'.format(name), 'r')

    keys = list(f.keys())
    for key in keys:
        print(key)
        group = f[key]
        for i in group.keys():
            print(group[i].shape)
    
def write_file(name):
    with h5py.File('{}'.format(name), 'w') as f:
        group = f.create_group('dataset1')
        for i in range(10):
            group[str(i)] = i

if __name__ == '__main__':
    name = 'test.hdf5'
    #write_file(name)
    read_file(name)