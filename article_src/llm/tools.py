def predict(row, pipe):
    sentence = row["sentence"]

    zero_shot_classification = pipe(
        sentence, candidate_labels=["Grammatical", "Ungrammatical"]
    )
    predicted_labels_numerical = [
        1 if prediction["labels"][0] == "Grammatical" else 0
        for prediction in zero_shot_classification
    ]
    return {"prediction": predicted_labels_numerical}
