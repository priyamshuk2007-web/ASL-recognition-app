import copy
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


def extract_geometric_features(X_63):
    """
    Takes (N, 63) normalized landmarks and appends 14 relational distance features.
    Returns (N, 77).
    """
    pts = X_63.reshape(-1, 21, 3)

    def dist(i, j):
        return np.linalg.norm(pts[:, i] - pts[:, j], axis=1, keepdims=True)

    thumb_to_tips = [dist(4, idx) for idx in [8, 12, 16, 20]]
    thumb_to_pips = [dist(4, idx) for idx in [6, 10, 14, 18]]
    finger_spread = [dist(8, 12), dist(12, 16)]

    curl_ratios = []
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        curl_ratios.append(dist(tip, 0) / (dist(mcp, 0) + 1e-6))

    extra = np.hstack(thumb_to_tips + thumb_to_pips + finger_spread + curl_ratios)
    return np.hstack([X_63, extra])


def augment_landmarks(X_63, y, rng):
    """
    Makes 3 extra copies of each training sample (4x total):
      rot1 (small rotation), mirror (flip left/right), rot2 (bigger rotation)
    """
    pts = X_63.reshape(-1, 21, 3)
    X_aug_list = [X_63]
    y_aug_list = [y]

    max_angle = {'rot1': 0.12, 'mirror': 0.12, 'rot2': 0.35}  # radians

    for mode in ['rot1', 'mirror', 'rot2']:
        p = pts.copy()

        if mode == 'mirror':
            p[:, :, 0] = -p[:, :, 0]

        a = max_angle[mode]
        angles = rng.uniform(-a, a, size=(len(p), 3))
        cx, cy, cz = np.cos(angles[:, 0]), np.cos(angles[:, 1]), np.cos(angles[:, 2])
        sx, sy, sz = np.sin(angles[:, 0]), np.sin(angles[:, 1]), np.sin(angles[:, 2])

        Rx = np.stack([
            np.stack([np.ones_like(cx), np.zeros_like(cx), np.zeros_like(cx)], axis=1),
            np.stack([np.zeros_like(cx), cx, -sx], axis=1),
            np.stack([np.zeros_like(cx), sx, cx], axis=1),
        ], axis=1)
        Ry = np.stack([
            np.stack([cy, np.zeros_like(cy), sy], axis=1),
            np.stack([np.zeros_like(cy), np.ones_like(cy), np.zeros_like(cy)], axis=1),
            np.stack([-sy, np.zeros_like(cy), cy], axis=1),
        ], axis=1)
        Rz = np.stack([
            np.stack([cz, -sz, np.zeros_like(cz)], axis=1),
            np.stack([sz, cz, np.zeros_like(cz)], axis=1),
            np.stack([np.zeros_like(cz), np.zeros_like(cz), np.ones_like(cz)], axis=1),
        ], axis=1)

        R = Rz @ Ry @ Rx
        p = np.einsum('nij,nkj->nki', R, p)
        p += rng.normal(0, 0.008, size=p.shape)

        X_aug_list.append(p.reshape(-1, 63))
        y_aug_list.append(y)

    return np.vstack(X_aug_list), np.concatenate(y_aug_list)


# 1. Load landmarks
data = pd.read_csv('landmarks.csv')

# 2. Random chunk split per letter: ~70% train, ~10% check, ~20% test
CHUNK = 30
split_rng = np.random.default_rng(7)
train_dfs, val_dfs, test_dfs = [], [], []

for letter, group in data.groupby('letter', sort=False):
    n_chunks = int(np.ceil(len(group) / CHUNK))
    order = split_rng.permutation(n_chunks)
    n_test = max(1, round(n_chunks * 0.2))
    n_val = max(1, round(n_chunks * 0.1))
    test_ids = set(order[:n_test])
    val_ids = set(order[n_test:n_test + n_val])
    for c in range(n_chunks):
        part = group.iloc[c * CHUNK:(c + 1) * CHUNK]
        if c in test_ids:
            test_dfs.append(part)
        elif c in val_ids:
            val_dfs.append(part)
        else:
            train_dfs.append(part)

# Make sure every letter has training data
train_letters = set(pd.concat(train_dfs)['letter'])
all_letters = set(data['letter'])
assert train_letters == all_letters, f"No training data for: {all_letters - train_letters}"


def to_xy(dfs):
    df = pd.concat(dfs, ignore_index=True)
    return df.drop('letter', axis=1).to_numpy(dtype=np.float32), df['letter'].to_numpy()


X_train_raw, y_train_raw = to_xy(train_dfs)
X_val_raw, y_val = to_xy(val_dfs)
X_test_raw, y_test = to_xy(test_dfs)

# 3. Make copies ONLY for the training set
rng = np.random.default_rng(42)
X_train_aug, y_train = augment_landmarks(X_train_raw, y_train_raw, rng)

# 4. Add distance features to all three sets
X_train = extract_geometric_features(X_train_aug)
X_val = extract_geometric_features(X_val_raw)
X_test = extract_geometric_features(X_test_raw)

print(f"Train: {len(X_train)}, Check: {len(X_val)}, Test: {len(X_test)}")

# 5. Scale, then train with our own early stopping
scaler = StandardScaler().fit(X_train)
Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                    alpha=1e-3, batch_size=256, random_state=42)
classes = np.unique(y_train)
best_acc, best_mlp, bad = 0, None, 0

for epoch in range(300):
    mlp.partial_fit(Xtr, y_train, classes=classes)
    acc = mlp.score(Xva, y_val)
    if acc > best_acc:
        best_acc, best_mlp, bad = acc, copy.deepcopy(mlp), 0
    else:
        bad += 1
        if bad >= 15:
            break

print("Stopped at epoch:", epoch + 1, "| Best check accuracy:", best_acc)

# 6. Final test
model = Pipeline([('scaler', scaler), ('mlp', best_mlp)])
predictions = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, predictions))
print("\nPer-letter breakdown:\n", classification_report(y_test, predictions))

# 7. Save model
joblib.dump(model, 'asl_letter_model.pkl')
print("Saved to asl_letter_model.pkl")