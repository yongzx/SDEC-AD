from models.SDEC_AD import DeepEmbeddingClustering
from scripts.load_data import load_data

if __name__ == "__main__":
    fn17_embedding_file = "data/lus_fn1.7_definition_bert.p"
    fnplus_embedding_file = "data/fnplus_bert_embed.p"

    newX, newY, num_frames, cut_off = load_data(fn17_embedding_file, fnplus_embedding_file)
    c = DeepEmbeddingClustering(n_clusters=num_frames,
                                input_dim=3072,
                                encoders_dims=[7500, 1000])

    ###### Training #####
    # autoencoder is only trained with LUs from BFN 1.7
    print("Train Autoencoder...")
    c.initialize(newX[:cut_off],
                 y=newY[:cut_off],
                 finetune_iters=100000,
                 layerwise_pretrain_iters=50000,
                 save_autoencoder=True)

    print("Clustering...")
    L = c.cluster(newX, y=newY, cut_off=cut_off, iter_max=1e6)