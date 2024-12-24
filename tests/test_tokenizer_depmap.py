import numpy as np
import os
import scanpy as sc

import sys
sys.path.append('/home/zinovyev/gits/micro-scgpt/src/')


from mscgpt.tokenizer import Tokenizer
from mscgpt import settings


def test_tokenizer_prepare_file():

    print(f'{settings.PATH_H5AD=}')

    # Data loading
    print('T Loading AnnData...')
    fname = "depmap.h5ad"
    adata = sc.read_h5ad(os.path.join(settings.PATH_H5AD, fname))
    adata = adata[np.logical_not(adata.obs['class'].isna())].copy()
    adata.var_names = adata.var_names.str.upper()

    # Quick data preprocessing
    print('T Preprocessing AnnData...')
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_counts=10)
    #sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.filter_cells(adata, min_counts=10)

    # Tokenizing
    print('T Tokenizing AnnData...')
    tk = Tokenizer()
    tokenized_dataset = tk.prepare_dataset(adata)
    tk.save_pretraining_dataset(tokenized_dataset, fname.split('.')[0])


def test_tokenizer_load_file_raw():

    print('T Loading file...')
    tk = Tokenizer()
    tokenized_dataset = tk.load_pretraining_dataset_raw("depmap")
    print(len(tokenized_dataset))
    print(tokenized_dataset[:5])


def test_tokenizer_load_bin_dataset():

    print('T Loading and binning file...')
    tk = Tokenizer()
    tokenized_dataset = tk.load_pretraining_dataset('depmap')
    print(len(tokenized_dataset))
    print(tokenized_dataset[:5])


if __name__ == "__main__":
    print(f'{__file__=}')
    print(f'{settings.PATH_ROOT=}')
    print('Run test_tokenizer_prepare_file...')
    test_tokenizer_prepare_file()
    print('test_tokenizer_load_file_raw...')
    test_tokenizer_load_file_raw()
    print('test_tokenizer_load_bin_dataset...')
    test_tokenizer_load_bin_dataset()
    pass