import dspy
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot
import random


def _manually_parse_output(output):
    try:
        narrative = output.split("Narrative: ")[1].split("\n")[0]
    except IndexError:
        print(f"Unable to parse output: {output}")
        return None
    # rationalization = output.split("Rationalization: ")[1].split("\n")[0]
    return dspy.Prediction(
        narrative=narrative,
        # rationalization=rationalization,
    )


class NarratorSig(dspy.Signature):
    """You are helping users understand an ML model's prediction. Given an explanation and information about the model,
    convert the explanation into a human-readable narrative."""

    context = dspy.InputField(desc="what the ML model predicts")
    explanation = dspy.InputField(desc="explanation of an ML model's prediction")
    explanation_format = dspy.InputField(desc="format the explanation is given in")

    narrative = dspy.OutputField(
        desc="human-readable narrative version of the explanation"
    )
    # rationalization = dspy.OutputField(
    #     desc="explains why given features may be relevant"
    # )


class Narrator:
    def __init__(
        self,
        explanation_format,
        context,
        llm=None,
        openai_api_key=None,
        sample_narratives=None,
        gpt_model_name="gpt-4o",
    ):
        """
        Args:
            explanation_format (string): Format explanations will take
            context (string): Brief description of what the model predicts (ie. "the model predicts house prices")
            llm (LLM object): DSPy LLM object to use.
                See https://dspy-docs.vercel.app/docs/building-blocks/language_models for examples
                One of llm or openai_api_key must be provided
            openai_api_key (string): OpenAI API key to use
            gpt_model_name (string): if openai_api_key is provided, specifies the GPT version to use
            sample_narratives (list of tuples of strings): List of (explanation, narrative) examples
        """
        self.llm = llm
        if self.llm is None and openai_api_key is not None:
            self.llm = dspy.OpenAI(
                model=gpt_model_name, api_key=openai_api_key, max_tokens=1000
            )
        self.context = context
        self.explanation_format = explanation_format
        self.sample_narratives = []
        if sample_narratives is not None:
            for example in sample_narratives:
                self.sample_narratives.append(
                    dspy.Example(
                        explanation=example[0],
                        narrative=example[1],
                        context=self.context,
                        explanation_format=explanation_format,
                    ).with_inputs("explanation", "context", "explanation_format")
                )

        self.few_shot_prompter = None
        self.bootstrapped_few_shot_prompter = None
        self.default_prompt = (
            "You are helping users understand an ML model's prediction. "
            "Given an explanation and information about the model, "
            "convert the explanation into a human-readable narrative."
        )

    def _assemble_prompt(
        self, prompt, explanation, explanation_format, examples=None, n=3
    ):
        header_string = f"{prompt}\n"
        format_string = (
            f"Follow the following format\n"
            f"Context: what the model predicts\n"
            f"Explanation: explanation of the model's prediction\n"
            f"Explanation Format: format the explanation is given in\n"
            f"Narrative: human-readable narrative version of the explanation\n"
        )
        input_string = (
            f"Context: {self.context}\n"
            f"Explanation: {explanation}\n"
            f"Explanation Format: {explanation_format}\n"
            "Please provide the output field Narrative. "
            "Do so immediately, without additional content before or after, "
            "and precisely as the format above shows."
        )

        examples_string = ""
        if examples is not None:
            for i, example in enumerate(random.sample(examples, n)):
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

    def narrate(self, explanation, n_examples=3, n_bootstrapped=0, grader=None):
        """
        Transform an explanation into a human-readable narrative

        Args:
            explanation (string): Explanation, in the format specified by self.explanation_format
            n_examples (int): Number of examples to pass
            n_bootstrapped (int): Number of bootstrapped examples to pass. Increasing this number
                will incur additional calls to the LLM, but may improve the quality of the output
                n_bootstrapped should be less than or equal to n_examples
            grader (Grader): Grader object to use for bootstrapping. Must be provided if n_bootstrapped > 0
        """
        if n_bootstrapped > 0:
            return self.bootstrap_few_shot(
                explanation,
                self.explanation_format,
                metric=grader,
                n_labeled_few_shot=n_examples,
                n_bootstrapped_few_shot=n_bootstrapped,
            ).narrative
        if self.sample_narratives:
            return self.few_shot(
                explanation, self.explanation_format, n_few_shot=n_examples
            ).narrative
        else:
            return self.basic_prompt(explanation, self.explanation_format).narrative

    def basic_prompt(self, explanation, explanation_format, prompt=None, few_shot_n=0):
        """
        Basic prompting

        Args:
            explanation (string): Explanation
            explanation_format (string): Explanation format
            prompt (string): Prompt
            few_shot_n (int): Number of examples to use in few-shot learning
        """
        if prompt is None:
            prompt = self.default_prompt
        full_prompt = self._assemble_prompt(
            prompt, explanation, explanation_format, examples=None
        )
        output = self.llm(full_prompt)[0]
        return _manually_parse_output(output)

    def few_shot(
        self, explanation, explanation_format, prompt=None, n_few_shot=3, use_dspy=False
    ):
        """
        Few-shot prompting

        Args:
            explanation (string): Explanation
            explanation_format (string): Explanation format
            prompt (string): Prompt
            n_few_shot (int): Number of examples to use in few-shot learning
            use_dspy (bool): Should be set to False, saving legacy version using DSPy in case needed later

        Returns:
            DSPy Prediction object
        """
        if prompt is None:
            prompt = self.default_prompt
        if not use_dspy:
            full_prompt = self._assemble_prompt(
                prompt,
                explanation,
                explanation_format,
                examples=self.sample_narratives,
                n=n_few_shot,
            )
            output = self.llm(full_prompt)[0]
            return _manually_parse_output(output)

    def bootstrap_few_shot(
        self,
        explanation,
        explanation_format,
        metric,
        n_labeled_few_shot=3,
        n_bootstrapped_few_shot=3,
    ):
        """
        Use DSPy to bootstrap few-shot prompts to optimize metrics

        Args:
            prompt (string): Not supported, included for consistency. To modify prompt, manually
                             edit the docstrings in the ExplingoSig object
            explanation (string): Explanation
            explanation_format (string): Explanation format
            metric (string): Metric to optimize
            n_labeled_few_shot (int): Number of examples to use in few-shot learning
            n_bootstrapped_few_shot (int): Number of bootstrapped examples to use in few-shot learning

        Returns:
            DSPy Prediction object
        """
        with dspy.context(lm=self.llm):
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=n_bootstrapped_few_shot,
                max_labeled_demos=n_labeled_few_shot,
                max_rounds=3,
            )
            self.bootstrapped_few_shot_prompter = optimizer.compile(
                dspy.Predict(NarratorSig),
                trainset=self.sample_narratives,
            )
            return self.bootstrapped_few_shot_prompter(
                explanation=explanation,
                explanation_format=explanation_format,
                context=self.context,
            )
