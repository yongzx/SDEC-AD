# SDEC-AD for Semantic Frame Induction

Keras implementation for our paper:
- Zheng-Xin Yong, Tiago Timponi Torrent. (2020). Semi-supervised Deep Embedded Clustering with Anomaly Detection for Semantic Frame Induction. In: Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020), Marseille, France.



## Usage
#### Dependencies
The dependencies are
- bcubed==1.5
- nltk==3.4.3
- matplotlib==3.2.1
- numpy==1.18.2
- Keras==2.2.5
- scikit_learn==0.23.0

Or simply, run `pip3 install -r requirements.txt` to install all the
dependencies.

#### Dataset Preparation
The data used in our research are as follows:
1. Berkeley FrameNet 1.7
2. FrameNet+
3. Curated anomalous lexical units (from WordNet). Can be accessed
through the LRE Map repository.

We use the Python `flair` library to generate the embeddings for the
lexical units using the exemplar sentences and their definitions.

The `data/` folder contains the embeddings of the lexical units.

#### Semantic Frame Induction
1. Create a folder `trained_SDEC_AD/` for saving the trained weights.
2. Run the Python script `python3 semantic_frame_induction_tr.py` to train the
SDEC-AD model.
3. Run the Python script `semantic_frame_induction_pred.py` to predict
and evaluate the clusters of LUs. Remember to update the parameter
 `SDEC_trained_weights` in the script to the trained weight that has the
 largest Bcubed F1-score (which is indicated in the name of the saved
 trained weights such as "SDEC_AD_bcubed_fscore_0.788.h5").

#### Anomalous Lexical Units Detection
1. Follow the instructions in the previous section "Semantic Frame
Induction" to generate the trained weights.
2. Update the parameter `SDEC_trained_weights` in the Python script
`anomaly_detection.py` to the trained weight that has the
 largest Bcubed F1-score. Then, run the Python script `anomaly_detection.py`
 to train the decoder and detect anomalous lexical units.
