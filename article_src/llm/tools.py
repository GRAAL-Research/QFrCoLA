def predict(row, pipe):
    sentence = row["sentence"]

    zero_shot_classification = pipe(
        sentence, candidate_labels=["Grammatical", "Ungrammatical"]
    )
    predicted_labels_numerical = []
    for prediction in zero_shot_classification:
        predicted_label = prediction["labels"][0]
        predicted_label_numerical = 1 if predicted_label == "Grammatical" else 0
        predicted_labels_numerical.append(predicted_label_numerical)
    return {"prediction": predicted_labels_numerical}
