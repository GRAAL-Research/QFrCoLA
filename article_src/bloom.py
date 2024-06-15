from transformers import pipeline

pipe = pipeline(task="zero-shot-classification", model="bigscience/bloom")

pipe(
    "I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["Grammatical", "Ungrammatical"],
)
