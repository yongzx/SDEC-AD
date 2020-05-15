from models.SDEC_AD import DeepEmbeddingClustering
from scripts.load_data import load_data
from evaluation.clustering import external_eval_clusters, print_external_eval_clusters

if __name__ == "__main__":
    fn17_embedding_file = "data/lus_fn1.7_definition_bert.p"
    fnplus_embedding_file = "data/fnplus_bert_embed.p"

    newX, newY, num_frames, cut_off = load_data(fn17_embedding_file, fnplus_embedding_file)
    c = DeepEmbeddingClustering(n_clusters=num_frames,
                                input_dim=3072,
                                encoders_dims=[7500, 1000])

    ###### Prediction #####
    print("Predicting...")
    pred_Y = c.predict(newX, SDEC_trained_weights=None)  # use the saved model when running the clustering (c.cluster)

    ###### Evaluation #####
    print("Evaluating...")
    print_external_eval_clusters(*external_eval_clusters(newY, pred_Y))