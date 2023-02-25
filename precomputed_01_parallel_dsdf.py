from dynasigml.dynasig_df import DynaSigDF
from multiprocessing import Pool
import glob
import numpy as np


def load_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    data_dict = dict()
    for line in lines[1:]:
        ll = line.split()
        data_dict[ll[2]] = [float(ll[0]), float(ll[1])]
    return data_dict


def run_dsdfs(data_dict, files, index):
    beta_values = [np.e ** (x / 2) for x in range(-6, 7)]
    exp_data = []
    for fn in files:
        mutid = fn.split('.')[0].split('mir125a_')[-1]
        exp_data.append(data_dict[mutid])
    DynaSigDF(files, exp_data, ["eff", "mcfold_energy"], "split_dsdfs/dsdf_{}".format(index), beta_values=beta_values)


def single_run(index):
    step = 300
    start = index*step
    stop = start+step
    data_dict = load_data("data_mir125.df")
    files = sorted(glob.glob('mir125a_variants/*.pdb'))
    if start < len(files):
        run_dsdfs(data_dict, files[start:stop], index)


def multiproc_run():
    indices = [x for x in range(99)]
    p = Pool(8)
    p.map(single_run, indices)


if __name__ == "__main__":
    multiproc_run()
