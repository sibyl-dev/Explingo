import dspy
import json


def create_example(convo):
    example = dspy.Example(
        explanation=convo["explanation"],
        context=convo["context"],
        explanation_format=convo["explanation_format"],
    )
    if "description" in convo:
        example.narrative = convo["description"]
    if "bad_description" in convo:
        example.bad_narrative = convo["bad_description"]
    return example.with_inputs("explanation", "context", "explanation_format")


def load_examples(json_file):
    training_data = json.load(open(json_file, "r"))
    examples = []
    for convo in training_data:
        examples.append(create_example(convo))
    return examples


def get_data(json_file, split=0.6):
    all_data = load_examples(json_file)
    labeled_data = [example for example in all_data if hasattr(example, "narrative")]
    unlabeled_data = [
        example for example in all_data if not hasattr(example, "narrative")
    ]

    labeled_train = labeled_data[: int(split * len(labeled_data))]
    labeled_eval = labeled_data[int(split * len(labeled_data)) :]
    unlabeled_train = unlabeled_data[: int(split * len(unlabeled_data))]
    unlabeled_eval = unlabeled_data[int(split * len(unlabeled_data)) :]

    return labeled_train, labeled_eval, unlabeled_train, unlabeled_eval
