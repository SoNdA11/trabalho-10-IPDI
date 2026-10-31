import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_extraction import extract_features, build_feature_names


def test_extract_features_shape():
    img = np.random.rand(64, 64, 3).astype(np.float32)
    fv = extract_features(img)
    assert fv.shape == (63,)
    assert np.isfinite(fv).all()


def test_feature_names_length():
    names = build_feature_names()
    assert len(names) == 63