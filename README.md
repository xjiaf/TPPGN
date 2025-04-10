# Trajectory Encoding Temporal Graph Networks [[arXiv](https:)]

## Running the experiments

### Requirements

### Dataset and Preprocessing
```{bash}
pip install -r requirements.txt
```

#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
* Wikipedia: http://snap.stanford.edu/jodie/wikipedia.csv
* Reddit: http://snap.stanford.edu/jodie/reddit.csv
* MOOC: http://snap.stanford.edu/jodie/mooc.csv
* LastFM: http://snap.stanford.edu/jodie/lastfm.csv

store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes
features are absent, they will be replaced by a vector of zeros.
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
python utils/preprocess_data.py --data mooc --bipartite
python utils/preprocess_data.py --data lastfm --bipartite
```

To produce feature-masked wikipedia or reddit dataset:
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite -fm
python utils/preprocess_data.py --data reddit --bipartite -fm
```

### Model Training

Self-supervised learning using the link prediction task:

Self-supervised learning on the wikipedia dataset (link prediction)
```{bash}
python train_self_supervised.py --use_memory --n_runs 2 --n_head 4 --n_layer 2 -pat "exp" --alpha 2 --beta 0.1 --scheduler 15 --gamma 0.1 --prefix "wikipedia_exp"
```

Self-supervised learning on the LastFM dataset
```{bash}
python train_self_supervised.py --data lastfm --use_memory --n_runs 2 --n_layer 2 -pat "exp" --alpha 1 --beta 0.1 -pd 20 -ped 60 --scheduler 20 --prefix "lastfm_exp"
```

Reddit
```{bash}
python train_self_supervised.py -d reddit --use_memory --n_runs 2 --n_layer 2 -pat "exp" --lr 0.00005 --scheduler 20 --gamma 0.33 --alpha 2 --beta 0.1 --prefix "reddit_exp"
```

Wiki-fm
```{bash}
python train_self_supervised.py -d wikipedia_fm --use_memory --n_runs 2 --n_head 4 --n_layer 2 -pat "exp" --alpha 1 --beta 0.1 -pd 20 -ped 60 --scheduler 15 --gamma 0.1 --prefix "wikipedia_fm_exp"
```

Reddit-fm
```{bash}
python train_self_supervised.py -d reddit_fm --use_memory --n_runs 2 --n_head 4 --n_layer 2 --lr 0.00005 --scheduler 20 --gamma 0.33 -pat "exp" --alpha 2 --beta 1 -pd 12 -ped 36 --prefix "reddit_fm_exp"
```
