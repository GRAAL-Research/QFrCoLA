from evaluate import load

ACCURACY = load("accuracy")
MCC = load("matthews_correlation")


def accuracy(predictions, references):
    return {
        key: round(value * 100, 2)
        for key, value in ACCURACY.compute(
            predictions=predictions, references=references
        ).items()
    }


def mcc(predictions, references):
    return {
        key: round(value, 3)
        for key, value in MCC.compute(
            predictions=predictions, references=references
        ).items()
    }
