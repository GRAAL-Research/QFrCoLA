def predict(row, pipe):
    sentence = row["sentence"]
    label = row["label"]
    zero_shot_classification = pipe(
        sentence, candidate_labels=["Grammatical", "Ungrammatical"]
    )
    predicted_label = zero_shot_classification["labels"][0]
    predicted_label_numerical = 1 if predicted_label == "Grammatical" else 0
    return {
        "zero_shot_classification": predicted_label_numerical,
        "good_prediction": predicted_label_numerical == label,
    }
