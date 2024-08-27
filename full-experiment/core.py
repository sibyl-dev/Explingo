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
    def __init__(
        self, llm, context, metric, labeled_train_data, unlabeled_train_data=None
    ):
        dspy.settings.configure(lm=llm, experimental=True)
        self.llm = llm
        self.context = context
        self.labeled_train_data = labeled_train_data
        self.unlabeled_train_data = (
            [] if unlabeled_train_data is None else unlabeled_train_data
        )
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

    def _run_experiment(self, func, eval_data, prompt=None, max_iters=100, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if prompt is None:
            prompt = (
                "You are helping users understand an ML model's prediction. Given an explanation and information about the model, convert the explanation into a human-readable narrative."
            )

        total_scores = None
        for i, example in enumerate(eval_data):
            if i >= max_iters:
                break
            result = func(
                prompt=prompt,
                explanation=example.explanation,
                explanation_format=example.explanation_format,
                **kwargs,
            )
            score = self.metric(example, result)
            if total_scores is None:
                total_scores = score[1]
            else:
                total_scores += score[1]

        total = min(max_iters, len(eval_data))
        average_scores = total_scores / total
        total_average_score = total_scores.sum() / total

        return total_average_score, average_scores

    def run_basic_prompting_experiment(self, eval_data, prompt=None, max_iters=100):
        """
        Run a basic prompting experiment

        Args:
            eval_data (list of DSPy examples): testing dataset
            prompt (string): Prompt
            max_iters (int): Maximum number of examples to run on

        Returns:
            total_average_score (float): Average total score over all explanations
            average_scores (pd.Series): Average scores for each metric
        """
        return self._run_experiment(
            self.basic_prompt,
            eval_data,
            prompt=prompt,
            max_iters=max_iters,
        )

    def run_few_shot_experiment(
        self, eval_data, prompt=None, max_iters=100, n_few_shot=3
    ):
        """
        Run a few-shot experiment

        Args:
            eval_data (list of DSPy examples): testing dataset
            prompt (string): Prompt
            max_iters (int): Maximum number of examples to run on
            n_few_shot (int): Number of examples to use in few-shot learning

        Returns:
            total_average_score (float): Average total score over all explanations
            average_scores (pd.Series): Average scores for each metric
        """
        return self._run_experiment(
            self.few_shot,
            eval_data,
            prompt=prompt,
            max_iters=max_iters,
            kwargs={"n_few_shot": n_few_shot},
        )

    def run_bootstrap_few_shot_experiment(
        self,
        eval_data,
        prompt=None,
        max_iters=100,
        n_labeled_few_shot=3,
        n_bootstrapped_few_shot=3,
    ):
        """
        Run a bootstrap few-shot experiment
        Args:
            eval_data (list of DSPy examples): testing dataset
            prompt (string): Prompt
            max_iters (int): Maximum number of examples to run on
            n_labeled_few_shot (int): Number of examples to use in few-shot learning
            n_bootstrapped_few_shot (int): Number of  bootstrapped examples to use in few-shot learning

        Returns:
            total_average_score (float): Average total score over all explanations
            average_scores (pd.Series): Average scores for each metric
        """
        return self._run_experiment(
            self.bootstrap_few_shot,
            eval_data,
            prompt=prompt,
            max_iters=max_iters,
            kwargs={
                "n_labeled_few_shot": n_labeled_few_shot,
                "n_bootstrapped_few_shot": n_bootstrapped_few_shot,
            },
        )

    def basic_prompt(self, prompt, explanation, explanation_format, few_shot_n=0):
        """
        Basic prompting


        """
        full_prompt = self.assemble_prompt(
            prompt, explanation, explanation_format, examples=None
        )
        output = self.llm(full_prompt)[0]
        return _manually_parse_output(output)

    def few_shot(
        self, prompt, explanation, explanation_format, n_few_shot=3, use_dspy=False
    ):
        """
        Few-shot prompting

        Args:
            prompt (string): Prompt
            explanation (string): Explanation
            explanation_format (string): Explanation format
            n_few_shot (int): Number of examples to use in few-shot learning
            use_dspy (bool): Should be set to False, saving legacy version using DSPy in case needed later

        Returns:
            DSPy Prediction object
        """
        if not use_dspy:
            full_prompt = self.assemble_prompt(
                prompt,
                explanation,
                explanation_format,
                examples=self.labeled_train_data,
            )
            output = self.llm(full_prompt)[0]
            return _manually_parse_output(output)
        if use_dspy:
            if self.few_shot_prompter is None:
                optimizer = LabeledFewShot(k=n_few_shot)
                self.few_shot_prompter = optimizer.compile(
                    dspy.Predict(ExplingoSig), trainset=self.labeled_train_data
                )
            return self.few_shot_prompter(
                explanation=explanation,
                explanation_format=explanation_format,
                context=self.context,
            )

    def bootstrap_few_shot(
        self,
        prompt,
        explanation,
        explanation_format,
        n_labeled_few_shot=3,
        n_bootstrapped_few_shot=3,
    ):
        """
        Use DSPy to bootstrap few-shot prompts to optimize metrics

        Args:
            prompt (string): Prompt
            explanation (string): Explanation
            explanation_format (string): Explanation format
            n_labeled_few_shot (int): Number of examples to use in few-shot learning
            n_bootstrapped_few_shot (int): Number of bootstrapped examples to use in few-shot learning

        Returns:
            DSPy Prediction object
        """
        optimizer = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=n_bootstrapped_few_shot,
            max_labeled_demos=n_labeled_few_shot,
        )
        self.bootstrapped_few_shot_prompter = optimizer.compile(
            dspy.Predict(ExplingoSig),
            trainset=self.labeled_train_data + self.unlabeled_train_data,
        )
        return self.bootstrapped_few_shot_prompter(
            explanation=explanation,
            explanation_format=explanation_format,
            context=self.context,
        )
