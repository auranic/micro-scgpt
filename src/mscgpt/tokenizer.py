import numpy as np
import os
import scanpy as sc
import time
import torch
from typing import List

from mscgpt import settings


class Tokenizer:
    """
    Manages data conversions between AnnData objects and
    tokenized inputs for model training.
    """
    def __init__(self, nbins: int = 5):
        # Loading already known gene ids
        self.nbins = nbins
        self.gid_path = settings.PATH_GIDS
        self.gtoi = {} # Gene name -> gene id
        self.itog = {} # Gene id -> gene name
        self._load_gene_ids()

    def _bin(self, x: torch.tensor) -> torch.tensor:
        x_out = torch.zeros_like(x)
        x_sorted, _ = torch.sort(x)
        step = x.shape[0] / self.nbins
        for i in range(self.nbins):
            x_out[x >= x_sorted[int(i*step)]] = i + 1
        return x_out
            
    def _load_gene_ids(self) -> None:
        # Loads gene ids from disk to memory
        try:
            f_gids = open(self.gid_path, "r")
        except IOError:
            raise FileNotFoundError(f"# Tokenizer: Cannot find {self.gid_path}. Terminating.")
        for line in f_gids:
            gname, gid = line.strip().split(",")
            self.gtoi[gname] = int(gid)
            self.itog[int(gid)] = gname
        print(f"> Tokenizer: {len(self.gtoi)} genes successfully loaded.")
        f_gids.close()

    def _update_gene_ids(self) -> None:
        # Writes gene ids from memory to disk
        with open(self.gid_path, "w") as f_out:
            f_out.write("\n".join(f"{gname},{gid}" for (gname, gid) in self.gtoi.items()))
        print(f"> Tokenizer: {len(self.gtoi)} genes successfully written.")

    def _get_gene_ids(self, gene_names: torch.tensor, add_if_not_found: bool = False) -> torch.tensor:
        ngenes = len(self.gtoi)
        result = []
        for gname in gene_names:
            gid = self.gtoi.get(gname, -1)
            if gid == -1 and add_if_not_found:
                self.gtoi[gname] = ngenes
                gid = ngenes
                ngenes += 1
            result.append(gid)
        return torch.tensor(result)

    def prepare_dataset(self, adata: sc.AnnData) -> List[torch.tensor]:
        # Preprocesses a single-cell dataset for pre-training.
        # Creates tokens corresponding to newly seen genes, 
        # Expects basic scRNA-seq preprocessing to already be performed.
        # Returns a list of N arrays of the shape 2T: [g_ids g_counts].
        step = adata.n_obs // 50
        samples = []
        tstart = time.time()
        for i, cell in enumerate(adata):
            nnz_idx = cell.X.nonzero()
            nnz_tokens = self._get_gene_ids(adata.var_names[nnz_idx[1]].to_numpy(),add_if_not_found=True)
            nnz_counts = torch.tensor(np.array([cell.X[nnz_idx]]).squeeze())
            samples.append(torch.cat((nnz_tokens, nnz_counts), axis=0))
            # Fancy progress bar
            if not i % step or i == adata.n_obs - 1:
                l = int(20 * i / adata.n_obs)
                bar = '=' * l + '.' * (20 - l)
                telapsed = time.time() - tstart
                print(f"> Processing dataset [{bar}] "
                      f"({str(i).rjust(len(str(adata.n_obs)))}/{adata.n_obs}) "
                      f"<{str(int(i/telapsed)).rjust(len(str(adata.n_obs)))} cells/s>", 
                      end='\r')
        print(f'\n> Terminated. Elapsed: {time.time() - tstart}s')
        self._update_gene_ids()
        return samples
    
    def save_pretraining_dataset(self, l: List[torch.tensor], fname: str) -> None:
        with open(os.path.join(settings.PATH_PRETRAIN, f"{fname}.tk"), 'w') as f_out:
            for x in l: 
                f_out.write(','.join([str(float(xi)) for xi in x]) + '\n')

    def load_pretraining_dataset_raw(self, fname: str) -> List[torch.tensor]:
        if not fname.endswith('.tk'):
            fname += '.tk'
        with open(os.path.join(settings.PATH_PRETRAIN, fname), 'r') as f_in:
            return [torch.tensor(list(map(float, row.strip().split(',')))) for row in f_in]
        
    def load_pretraining_dataset(
            self, 
            fname: str,
            nbins: int = 5
        )-> List[torch.tensor]:
        raw_data = self.load_pretraining_dataset_raw(fname)
        result = []
        for x in raw_data: #TODO: efficiency?
            x = x.view(2, -1)
            x[1] = self._bin(x[1])
            result.append(x.long())
        return result


if __name__ == "__main__":
    pass