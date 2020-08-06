import json

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize, differential_evolution
from seaborn import heatmap


def loss(weights, *args):
    preds = args[0].copy()
    trues = args[1]
    for i in range(len(weights)):
        preds[i] = preds[i] * weights[i]
    preds = preds.sum(0)

    roc_auc = roc_auc_score(trues, preds)
    return 1.0 - roc_auc


def search_weights(predicts, labels):
    bounds = [(0, 1)] * predicts.shape[0]
    weights = np.array([1 / predicts.shape[0]] * predicts.shape[0])

    res = minimize(
        loss,
        weights,
        method="L-BFGS-B",
        options={"maxiter": 1000, "eps": 1e-2},
        args=(predicts, labels),
        bounds=bounds,
    )
    weights = res["x"]
    for i in range(len(weights)):
        predicts[i] = predicts[i] * weights[i]
    join_predicts = predicts.sum(0)
    roc_auc = roc_auc_score(labels, join_predicts)
    print(f"Final weights - {weights}")
    print(f"Final score - {roc_auc}")
    return weights, roc_auc


# def search_weights(predicts, labels):
#     bounds = [(0, 4)] * predicts.shape[0]
#     weights = np.array([1 / predicts.shape[0]] * predicts.shape[0])
#
#     res = differential_evolution(
#         loss,
#         bounds=bounds,
#         args=(predicts, labels),
#         # bounds=bounds,
#     )
#     weights = res["x"]
#     for i in range(len(weights)):
#         predicts[i] = predicts[i] * weights[i]
#     join_predicts = predicts.sum(0)
#     roc_auc = roc_auc_score(labels, join_predicts)
#     print(f"Final weights - {weights}")
#     print(f"Final score - {roc_auc}")


if __name__ == "__main__":
    predicts = pd.read_csv("../models/predicts.csv")
    assert np.array_equal(
        predicts["target_256"].values, predicts["target_512"].values
    ), "FUCK OFF!"
    assert np.array_equal(
        predicts["target_768"].values, predicts["target_512"].values
    ), "FUCK OFF!"

    # predict_columns = list(filter(lambda x: "fold" in x, predicts.columns))
    predict_columns = [
        "fold_1_256_v1",
        "fold_2_256_v1",
        "fold_0_512_v1",
        "fold_3_512_v1",
        "fold_4_512_v1",
        "fold_1_768_v1",
        "fold_3_768_v1",
    ]
    probabilities = np.moveaxis(predicts[predict_columns].values, -1, 0)
    labels = predicts["target_256"].values

    weights, score = search_weights(probabilities.copy(), labels)

    with open("../models/weights.json", "w") as f:
        d = dict(
            final_score=score,
            weights={col: weights[idx] for idx, col in enumerate(predict_columns)},
        )
        json.dump(d, f)

    print(
        f"Score by median averaging - {roc_auc_score(labels, np.median(probabilities, axis=0))}"
    )
    print(
        f"Score by mean averaging - {roc_auc_score(labels, np.mean(probabilities, axis=0))}"
    )

    predictions = np.zeros_like(labels)
    for predict in probabilities:
        predictions = np.add(predictions, rankdata(predict) / predictions.shape[0])
    predictions /= len(probabilities)
    print(f"Score by rank averaging - {roc_auc_score(labels, predictions)}")

    corr = np.corrcoef(probabilities)
    ax = heatmap(corr, xticklabels=predict_columns)
    ax.figure.savefig("../models/correlation_matrix.png")
