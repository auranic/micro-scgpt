import sys
sys.path.append('/home/zinovyev/gits/micro-scgpt/src/')

from mscgpt.data_handler import DataHandler
from mscgpt.tokenizer import Tokenizer

def test_load_data_handler():
    print('T Creating Tokenizer')
    tk = Tokenizer()
    data = tk.load_pretraining_dataset('depmap')
    print('T Loading dataset in data handler')
    dh = DataHandler(32, 500, tk.pad_token, val_ratio=0.15, test_ratio=0.05)
    dh.load_dataset(data)
    print(dh)

def test_get_batch_data_handler():
    print('T Creating Tokenizer')
    tk = Tokenizer()
    data = tk.load_pretraining_dataset('depmap')
    print('T Loading dataset in data handler')
    dh = DataHandler(32, 500, tk.pad_token, val_ratio=0.15, test_ratio=0.05)
    dh.load_dataset(data)
    print('T Geting batch')
    gids, bins, cnts = dh.get_batch('train')
    print(gids.shape, bins.shape, cnts.shape)
    print(gids[0])
    print(bins[0])
    print(cnts[0])

if __name__ == "__main__":
    test_load_data_handler()
    test_get_batch_data_handler()