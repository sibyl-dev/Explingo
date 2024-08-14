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
