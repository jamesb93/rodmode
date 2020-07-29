from flucoma import fluid
from flucoma.utils import get_buffer
from ftis.common.io import write_json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

testers = [x for x in Path("Training").iterdir()]

data = []
js = {}

for x in testers:
    mfcc = get_buffer(
        fluid.stats(
            fluid.mfcc(x, numcoeffs=20, fftsettings=[256, 64, 512]),
            numderivs=1
        ), "numpy"
    ).flatten()

    data.append(mfcc)

data = np.array(data)

# scaler = StandardScaler().fit(data)
# scaled = scaler.transform(data)
# joblib.dump(scaler, "standardisation.joblib")

for name, points in zip(testers, data):
    js[str(name)] = list(points)

write_json("trainingmfcc.json", js)