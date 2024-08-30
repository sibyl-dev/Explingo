import dspy
import json
import random


def create_example(entry):
    example = dspy.Example(
        explanation=entry["explanation"],
        context=entry["context"],
        explanation_format=entry["explanation_format"],
    )
    if "narrative" in entry:
        example.narrative = entry["narrative"]
    if "bad_narrative" in entry:
        example.bad_narrative = entry["bad_narrative"]
    return example.with_inputs("explanation", "context", "explanation_format")


def load_examples(json_file):
    training_data = json.load(open(json_file, "r"))
    examples = []
    for entry in training_data:
        examples.append(create_example(entry))
    return examples


def get_data(json_file, split=None):
    all_data = load_examples(json_file)
    labeled_data = [example for example in all_data if hasattr(example, "narrative")]
    unlabeled_data = [
        example for example in all_data if not hasattr(example, "narrative")
    ]
    if split is not None:
        labeled_train = labeled_data[: int(split * len(labeled_data))]
        labeled_eval = labeled_data[int(split * len(labeled_data)) :]
        unlabeled_train = unlabeled_data[: int(split * len(unlabeled_data))]
        unlabeled_eval = unlabeled_data[int(split * len(unlabeled_data)) :]
    else:
        labeled_train = labeled_data[:5]
        labeled_eval = labeled_data[5:]
        unlabeled_train = unlabeled_data[:5]
        unlabeled_eval = unlabeled_data[5:]
        if len(unlabeled_train) < 5:
            additional_count = 5 - len(unlabeled_train)
            labeled_train += labeled_eval[:additional_count]
            labeled_eval = labeled_eval[additional_count:]

    return labeled_train, labeled_eval, unlabeled_train, unlabeled_eval
