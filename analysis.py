# from ftis.analyser import FluidMFCC, Stats
# from ftis.process import FTISProcess
from flucoma import fluid
from flucoma.utils import get_buffer
from pathlib import Path
from umap import UMAP
from joblib import dump, load
from sklearn.neighbors import KDTree
from utils import playback
from ftis.common.io import read_json
import numpy as np
import simpleaudio as sa
import math
from pydub import AudioSegment
training = [x for x in Path("Training").iterdir()]
testers = [x for x in Path("Testing").iterdir()]

values, keys = [], []
fft = [256, 64, 512]
coefs = 20
derivs = 1
training_data = read_json("trainingmfcc.json")
testing_data = read_json("testingmfcc.json")

for x in training:
    keys.append(str(x))

if not Path("embedding.joblib").exists() or not Path("tree.joblib").exists():

    keys = [k for k in training_data.keys()]
    values = [v for v in training_data.values()]

    data = np.array(values)
    embedding = UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.000,
        random_state=42
    ).fit(data)

    redux = embedding.transform(data)
    tree = KDTree(redux, leaf_size=2)

    dump(tree, 'tree.joblib')
    dump(embedding, 'embedding.joblib')

embedding = load("embedding.joblib")
tree = load("tree.joblib")

z = 0
container = AudioSegment.empty()
for x in testing_data:
    val = testing_data[x]
    trans = embedding.transform([val])
    _, ind = tree.query(trans, k=4)
    # _, ind = tree.query([val], k=4)

    # match = keys[ind[0][0]]
    # testing_num = int(str(x)[26:-12])
    # training_num = int(str(match)[27:-12])
    for y in range(4):
        playback(keys[ind[0][y]])
    
    # if testing_num == training_num:
        # z += 1
    # r = math.ceil(testing_num / 5)
    # print(r, training_num)

# print(z / len(testing_data) * 100.0)