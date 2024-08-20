import dspy
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot
import random


def _manually_parse_output(output):
    narrative = output.split("Narrative: ")[1].split("\n")[0]
    rationalization = output.split("Rationalization: ")[1].split("\n")[0]
    return dspy.Prediction(
        narrative=narrative,
        rationalization=rationalization,
    )


class ExplingoSig(dspy.Signature):
    """You are helping users understand an ML model's prediction. Given an explanation and information about the model,
    convert the explanation into a human-readable narrative."""

    context = dspy.InputField(desc="what the ML model predicts")
    explanation = dspy.InputField(desc="explanation of an ML model's prediction")
    explanation_format = dspy.InputField(desc="format the explanation is given in")

    narrative = dspy.OutputField(
        desc="human-readable narrative version of the explanation"
    )
    rationalization = dspy.OutputField(
        desc="explains why given features may be relevant"
    )


class Explingo:
    def __init__(self, llm, context, examples, metric):
        dspy.settings.configure(lm=llm, experimental=True)
        self.llm = llm
        self.context = context
        self.examples = examples
        self.few_shot_prompter = None
        self.bootstrapped_few_shot_prompter = None
        self.metric = metric

    def assemble_prompt(
        self, prompt, explanation, explanation_format, examples=None, k=3
    ):
        header_string = f"{prompt}\n"
        format_string = (
            f"Follow the following format\n"
            f"Context: what the model predicts\n"
            f"Explanation: explanation of the model's prediction\n"
            f"Explanation Format: format the explanation is given in\n"
            f"Narrative: human-readable narrative version of the explanation\n"
            f"Rationalization: explains why given features may be relevant\n"
        )
        input_string = (
            f"Context: {self.context}\n"
            f"Explanation: {explanation}\n"
            f"Explanation Format: {explanation_format}\n"
            "Please provide the output fields Narrative then Rationalization. "
            "Do so immediately, without additional content before or after, "
            "and precisely as the format above shows. Begin with the field Narrative."
        )

        examples_string = ""
        if examples is not None:
            for i, example in enumerate(random.sample(examples, k)):
                examples_string += (
                    f"Example {i+1}\n"
                    f"Context: {example.context}\n"
                    f"Explanation: {example.explanation}\n"
                    f"Explanation Format: {example.explanation_format}\n"
                    f"Narrative: {example.narrative}\n"
                )

        if len(examples_string) == 0:
            return "---\n".join([header_string, format_string, input_string])
        else:
            return "---\n".join(
                [header_string, format_string, examples_string, input_string]
            )

    def run_experiment(
        self,
        dataset,
        prompt_type="basic",
        prompt="You are helping users understand an ML model's prediction. Given an explanation and information about the model, convert the explanation into a human-readable narrative.",
        max_iters=100,
    ):
        """
        :param dataset: List of example objects to evaluate on
        :param prompt_type: One of "basic", "few-shot", "bootstrap-few-shot"
        :param prompt: Currently unused
        :param max_iters: Maximum number of explanations to evaluate on
        :return: Average total score over all explanations (currently 0-8)
        """
        if prompt_type == "basic":
            func = self.prompt
        elif prompt_type == "few-shot":
            func = self.few_shot
        elif prompt_type == "bootstrap-few-shot":
            func = self.bootstrap_few_shot
        else:
            raise ValueError(
                "Invalid type. Options: basic, few-shot, bootstrap-few-shot"
            )

        total_score = 0
        all_scores = None
        total_count = 0
        for example in dataset:
            result = func(
                prompt=prompt,
                explanation=example.explanation,
                explanation_format=example.explanation_format,
            )
            score = self.metric(example, result)
            total_score += score[0]
            total_count += 1
            if all_scores is None:
                all_scores = score[1]
            else:
                all_scores += score[1]
            if total_count >= max_iters:
                break
        return total_score / total_count, all_scores / total_count

    def prompt(self, prompt, explanation, explanation_format):
        full_prompt = self.assemble_prompt(
            prompt, explanation, explanation_format, examples=None
        )
        output = self.llm(full_prompt)[0]
        return _manually_parse_output(output)

    def few_shot(self, prompt, explanation, explanation_format, use_dspy=False):
        if not use_dspy:
            examples_with_labels = [
                example
                for example in self.examples
                if hasattr(example, "narrative") and example.narrative is not None
            ]
            full_prompt = self.assemble_prompt(
                prompt, explanation, explanation_format, examples=examples_with_labels
            )
            output = self.llm(full_prompt)[0]
            return _manually_parse_output(output)
        if use_dspy:
            if self.few_shot_prompter is None:
                optimizer = LabeledFewShot(k=3)
                self.few_shot_prompter = optimizer.compile(
                    dspy.Predict(ExplingoSig), trainset=self.examples
                )
            return self.few_shot_prompter(
                explanation=explanation,
                explanation_format=explanation_format,
                context=self.context,
            )

    def bootstrap_few_shot(self, prompt, explanation, explanation_format):
        if self.bootstrapped_few_shot_prompter is None:
            optimizer = BootstrapFewShot(metric=self.metric)
            self.bootstrapped_few_shot_prompter = optimizer.compile(
                dspy.Predict(ExplingoSig), trainset=self.examples
            )
        return self.bootstrapped_few_shot_prompter(
            explanation=explanation,
            explanation_format=explanation_format,
            context=self.context,
        )
