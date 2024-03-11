import torch

from typing import Literal, Tuple

class DataHandler:
    """
    The data handler shapes the input data to
    feed the model. 
    """
    def __init__(
            self, 
            batch_size: int, 
            ctx_size: int, 
            pad_token: int,
            val_ratio: float = .15, 
            test_ratio: float = .05,
            device: str = "cpu"
        ):
        self.batch_size = batch_size
        self.ctx_size = ctx_size
        self.pad_token = pad_token
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.device = device

        # Gene id tokens
        self.data = {
            "train": {
                "gid": torch.zeros((0, 0)), # Will grow dynamically
                "bin": torch.zeros((0, 0)),
                "cnt": torch.zeros((0, 0))
            },
            "val": {
                "gid": torch.zeros((0, 0)),
                "bin": torch.zeros((0, 0)),
                "cnt": torch.zeros((0, 0))
            },
            "test": {
                "gid": torch.zeros((0, 0)),
                "bin": torch.zeros((0, 0)),
                "cnt": torch.zeros((0, 0))
            }
        }

    def __repr__(self) -> str:
        s = ""
        for mkey, mitem in self.data.items():
            for dkey, ditem in mitem.items():
                s += f"[{mkey},{dkey}]\t{ditem.shape}\t{ditem[0, :10]}\n"
        return s[:-1]
        
    def _resize_data(self, d: int) -> None:
        # Dynamically grows data tensors if necessary
        # TODO: could be done in a smarter way
        old_size = self.data["train"]["gid"].shape[1]
        for mkey, mitem in self.data.items():
            for dkey, ditem in mitem.items():
                new_x = torch.zeros((ditem.shape[0], d)) + self.pad_token
                new_x[:, :old_size] = ditem
                self.data[mkey][dkey] = new_x

    def load_dataset(self, data: list[torch.tensor]) -> None:
        """
        TODO
        """
        # Dynamic resizing
        max_shape_1 = max(x.shape[1] for x in data)
        if max_shape_1 > self.data["train"]["gid"].shape[1]:
            self._resize_data(max_shape_1)

        # Dataset splicing
        n = len(data)
        nval = int(self.val_ratio * n)
        ntest = int(self.test_ratio * n)
        ntrain = n - nval - ntest
        new_data = {
            "train": {
                "gid": torch.zeros((ntrain, max_shape_1)) + self.pad_token, # Will grow dynamically
                "bin": torch.zeros((ntrain, max_shape_1)),
                "cnt": torch.zeros((ntrain, max_shape_1))
            },
            "val": {
                "gid": torch.zeros((nval, max_shape_1)) + self.pad_token,
                "bin": torch.zeros((nval, max_shape_1)),
                "cnt": torch.zeros((nval, max_shape_1))
            },
            "test": {
                "gid": torch.zeros((ntest, max_shape_1)) + self.pad_token,
                "bin": torch.zeros((ntest, max_shape_1)),
                "cnt": torch.zeros((ntest, max_shape_1))
            }
        }
        for i, x in enumerate(data): # [3, d]
            if i < ntrain:
                off = 0
                m = "train"
            elif ntrain <= i < ntrain + nval:
                off = ntrain
                m = "val"
            else:
                off = ntrain + nval
                m = "test"
            new_data[m]["gid"][i - off, :x.shape[1]] = x[0]
            new_data[m]["cnt"][i - off, :x.shape[1]] = x[1]
            new_data[m]["bin"][i - off, :x.shape[1]] = x[2]

        # Appending new data
        for dkey, ditem in self.data.items():
            for mkey, mitem in ditem.items(): 
                self.data[dkey][mkey] = torch.cat((self.data[dkey][mkey], new_data[dkey][mkey]), axis=0)

        print('> Data Handler: Dataset successfully loaded.')

    def get_batch(self, mode: Literal["train", "val", "test"]) -> Tuple[torch.tensor]:
        """
        TODO
        """
        assert mode in ["train", "val", "test"], (
            f"Unknown mode: {mode}. "
            "Expected 'train', 'val' or 'test'."
        )
        data = self.data[mode]
        n, d = data["gid"].shape[0], data["gid"].shape[1]
        idx = torch.randint(n, (self.batch_size,))
        gids = torch.zeros((self.batch_size, self.ctx_size)).type(torch.long) + self.pad_token
        bins = torch.zeros((self.batch_size, self.ctx_size)).type(torch.long)
        cnts = torch.zeros((self.batch_size, self.ctx_size)).type(torch.long)
        gids[:, :d] = data["gid"][idx]
        bins[:, :d] = data["bin"][idx]
        cnts[:, :d] = data["cnt"][idx]
        tkperm = torch.randperm(self.ctx_size)
        gids = gids[:, tkperm]
        bins = bins[:, tkperm]
        cnts = cnts[:, tkperm]
        return gids.to(self.device), bins.to(self.device), cnts.to(self.device)