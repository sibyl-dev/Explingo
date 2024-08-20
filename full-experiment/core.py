import dspy
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot
import numpy as np


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
    def __init__(self, context, examples, metric):
        self.context = context
        self.examples = examples
        self.few_shot_prompter = None
        self.bootstrapped_few_shot_prompter = None
        self.metric = metric

    def run_experiment(
        self,
        explanations,
        explanation_format,
        prompt_type="basic",
        prompt=None,
        max_iters=100,
    ):
        """
        TODO: also return average score over each individual metric
        :param explanations: List of evaluation explanations
        :param explanation_format: Format of the explanations
        :param type: One of "basic", "few-shot", "bootstrap-few-shot"
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
        for exp in explanations:
            result = func(explanation=exp, explanation_format=explanation_format)
            score = self.metric(
                dspy.Example(explanation=exp, explanation_format=explanation_format),
                result,
            )
            total_score += score[0]
            total_count += 1
            if all_scores is None:
                all_scores = score[1]
            else:
                all_scores += score[1]
            if total_count >= max_iters:
                break
        return total_score / total_count, all_scores / total_count

    def prompt(self, explanation, explanation_format):
        return dspy.Predict(ExplingoSig)(
            explanation=explanation,
            explanation_format=explanation_format,
            context=self.context,
        )

    def few_shot(self, explanation, explanation_format):
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

    def bootstrap_few_shot(self, explanation, explanation_format):
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
