import numpy as np
from sklearn.metrics import make_scorer, matthews_corrcoef


def score_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


score_mcc_ = make_scorer(score_mcc, greater_is_better=True)


# Print summary statistics of the results
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")
