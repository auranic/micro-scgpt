from mscgpt.tokenizer import Tokenizer
from mscgpt.data_handler import DataHandler
from mscgpt.gpt import MicroSCGPT, GeneExpressionRegressor

def test_forward():
    print('T Creating Tokenizer')
    tk = Tokenizer()
    data = tk.load_pretraining_dataset('tabmuris_A')
    print('T Loading dataset in data handler')
    dh = DataHandler(32, 500, tk.pad_token, val_ratio=0.15, test_ratio=0.05)
    dh.load_dataset(data)
    print('T Geting batch')
    gids, bins, cnts = dh.get_batch('train')
    print('T Forward pass')
    model = MicroSCGPT(dh.ctx_size, tk.bins_size, tk.vocab_size)
    output = model(gids, bins)
    print(output.shape)

def test_forward_gene_expr_prediction():
    print('T Creating Tokenizer')
    tk = Tokenizer()
    data = tk.load_pretraining_dataset('tabmuris_A')
    print('T Loading dataset in data handler')
    dh = DataHandler(8, tk.n_genes, tk.pad_token, val_ratio=0.15, test_ratio=0.05)
    dh.load_dataset(data)
    print('T Geting batch')
    gids, bins, cnts = dh.get_batch('train')
    print('T Forward pass')
    model = MicroSCGPT(dh.ctx_size, tk.bins_size, tk.vocab_size)
    output = model(gids, bins)
    print('T Prediction')
    model_gexpr = GeneExpressionRegressor(dh.ctx_size, model.embed_size, tk.n_genes)
    genes_pred = model_gexpr(output)
    print(genes_pred.shape)
    print(genes_pred[0])

if __name__ == "__main__":
    test_forward_gene_expr_prediction()