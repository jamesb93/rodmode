from flucoma import fluid
from flucoma.utils import get_buffer
from ftis.common.io import write_json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

testers = [x for x in Path("Testing").iterdir()]

data = []
js = {}

for x in testers:
    mfcc = get_buffer(
        fluid.stats(
            fluid.mfcc(x, numcoeffs=20, fftsettings=[256, 64, 512]),
            numderivs=1
        ), "numpy"
    ).flatten()

    js[str(x)] = list(mfcc)

write_json("testingmfcc.json", js)
