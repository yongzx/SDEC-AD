# from keras_sdec import DeepEmbeddingClustering
from models.SDEC_AD import DeepEmbeddingClustering
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scripts.load_data import load_data, load_anomalous_synsets
import numpy as np

def plot_roc_pr(fpr_sdec, tpr_sdec, roc_thresholds_sdec, roc_auc_sdec,
                prec_sdec, recall_sdec, pr_thresholds_sdec, pr_auc_sdec):
    def adjust_spines(ax, spines):
        for loc, spine in ax.spines.items():
            if loc in spines:
                spine.set_position(('outward', 10))  # outward by 10 points
                spine.set_smart_bounds(True)
            else:
                spine.set_color('none')  # don't draw spine

        # turn off ticks where there is no spine
        if 'left' in spines:
            ax.yaxis.set_ticks_position('left')
        else:
            # no yaxis ticks
            ax.yaxis.set_ticks([])

        if 'bottom' in spines:
            ax.xaxis.set_ticks_position('bottom')
        else:
            # no xaxis ticks
            ax.xaxis.set_ticks([])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label_font = {"fontname": "Arial", "fontsize": 10}

    fig, ax1 = plt.subplots()
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', **label_font)
    ax1.set_ylabel('True Positive Rate', **label_font)

    # plot ROC results
    ax1.plot(fpr_sdec, tpr_sdec, clip_on=False, color='#d7191c', lw=1.3, alpha=0.75,
             ls="-", label='SDEC-AD (AUC = %0.2f)' % roc_auc_sdec)
    adjust_spines(ax1, ['left', 'bottom'])

    # display only a left and bottom box border in matplotlib
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.legend(loc="lower right", bbox_to_anchor=(1.05, -0.025), handlelength=3.5, borderpad=1, labelspacing=1,
               prop={"family": "Arial", "size": 9.5})
    fig.savefig('roc.eps', format='eps')
    fig.savefig('roc.png', format='png')
    plt.close(fig)

    ### plot PRC
    fig, ax2 = plt.subplots()
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', **label_font)
    ax2.set_ylabel('Precision', **label_font)

    # plot PRC results
    ax2.plot(recall_sdec, prec_sdec, clip_on=False, color='#d7191c', lw=1.3, alpha=0.75,
             ls="-", label='SDEC-AD (AUC = %0.2f)' % pr_auc_sdec)
    adjust_spines(ax2, ['left', 'bottom'])

    ax2.legend(loc="upper right", handlelength=3.5, borderpad=1.2, labelspacing=1.2,
               prop={"family": "Arial", "size": 9.5})
    fig.savefig('prc.eps', format='eps')
    fig.savefig('prc.png', format='png')
    plt.close(fig)

if __name__ == "__main__":
    fn17_embedding_file = "data/lus_fn1.7_definition_bert.p"
    fnplus_embedding_file = "data/fnplus_bert_embed.p"
    anomalous_synset_embedding_filename = "data/anomalies_definition_bert.p"

    newX, newY, num_frames, cut_off = load_data(fn17_embedding_file, fnplus_embedding_file)
    c = DeepEmbeddingClustering(n_clusters=num_frames,
                                input_dim=3072,
                                encoders_dims=[7500, 1000])

    # print("Train Autoencoder...")
    # c.initialize(newX[:cut_off],
    #              y=newY[:cut_off],
    #              finetune_iters=100000,
    #              layerwise_pretrain_iters=50000,
    #              save_autoencoder=True)
    #
    # print("Clustering...")
    # L = c.cluster(newX, y=newY, cut_off=cut_off, iter_max=1e6)

    print("Train Decoder...")
    c.train_decoders(SDEC_trained_weights=...,  # use the saved model when running the clustering (c.cluster)
                     X=newX,
                     finetune_iters=100000,
                     layerwise_pretrain_iters=50000)

    print("Anomaly Detection...")
    anom_X, anom_Y = load_anomalous_synsets(anomalous_synset_embedding_filename)
    combined_X = np.concatenate((newX, anom_X), axis=0)
    combined_Y = np.concatenate((np.zeros(shape=newX.shape[0]), np.ones(shape=anom_X.shape[0])), axis=0)

    norm_scores = c.reconstruction_loss(newX, individual=True)
    anom_scores = c.reconstruction_loss(anom_X, individual=True)
    scores = np.concatenate((norm_scores, anom_scores), axis=0)
    y_labels = np.concatenate((np.zeros(shape=newX.shape[0]), np.ones(shape=anom_X.shape[0])), axis=0)

    fpr, tpr, roc_thresholds = roc_curve(y_labels, scores, pos_label=1)
    prec, recall, pr_thresholds = precision_recall_curve(y_labels, scores, pos_label=1)
    roc_auc_sdec = auc(fpr, tpr)
    pr_auc_sdec = auc(recall, prec)
    plot_roc_pr(fpr, tpr, roc_thresholds, roc_auc_sdec, prec, recall, pr_thresholds, pr_auc_sdec)



