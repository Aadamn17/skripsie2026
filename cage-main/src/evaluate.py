import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def extract_features(encoder, loader, device='cuda'):
    encoder.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for view1, view2, labels in loader:
            view1 = view1.to(device).float()
            feat = encoder(view1).cpu().numpy()
            features_list.append(feat)
            labels_list.append(labels.numpy())
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


def evaluate_classifier(encoder, train_loader, test_loader, device='cuda'):
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)

    if len(np.unique(train_labels)) < 2:
        return 0.5, 0.5

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    probs = clf.predict_proba(test_features)[:, 1]

    acc = accuracy_score(test_labels, preds)
    auc = roc_auc_score(test_labels, probs) if len(np.unique(test_labels)) == 2 else 0.5
    return acc, auc