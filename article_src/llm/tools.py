def predict(row, pipe):
    sentence = row["sentence"]

    zero_shot_classification = pipe(
        sentence, candidate_labels=["Grammatical", "Ungrammatical"]
    )
    predicted_label = zero_shot_classification["labels"][0]
    predicted_label_numerical = 1 if predicted_label == "Grammatical" else 0
    return {"prediction": predicted_label_numerical}
