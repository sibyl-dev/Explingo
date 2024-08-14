import dspy
from metrics import all_metrics
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot


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
    def __init__(self, context, examples):
        self.context = context
        self.examples = examples
        self.few_shot_prompter = None
        self.bootstrapped_few_shot_prompter = None
        self.metric = all_metrics

    def run_experiment(
        self, explanations, explanation_format, type="basic", prompt=None, max_iters=100
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
        if type == "basic":
            func = self.prompt
        elif type == "few-shot":
            func = self.few_shot
        elif type == "bootstrap-few-shot":
            func = self.bootstrap_few_shot
        else:
            raise ValueError(
                "Invalid type. Options: basic, few-shot, bootstrap-few-shot"
            )

        score = 0
        total = 0
        for exp in explanations:
            result = func(explanation=exp, explanation_format=explanation_format)
            score += self.metric(
                dspy.Example(explanation=exp, explanation_format=explanation_format),
                result,
            )
            total += 1
            if total >= max_iters:
                break
        return score / len(explanations)

    def prompt(self, explanation, explanation_format):
        return dspy.Predict(ExplingoSig)(
            explanation=explanation,
            explanation_format=explanation_format,
            context=self.context,
        )

    def few_shot(self, explanation, explanation_format):
        if self.few_shot_prompter is None:
            optimizer = LabeledFewShot()
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
