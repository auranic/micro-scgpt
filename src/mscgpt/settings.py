import os

# Files
PATH_ROOT = os.path.join(*__file__.split(os.sep)[:-3])
if __file__[0] == os.sep: PATH_ROOT = os.sep + PATH_ROOT
PATH_DATA = os.path.join(PATH_ROOT, "data")
PATH_GIDS = os.path.join(PATH_DATA, "assets", "gene_ids.csv")
PATH_H5AD = os.path.join(PATH_DATA, "h5ad")
PATH_PRETRAIN = os.path.join(PATH_DATA, "pretrain")