import dspy
import pandas as pd
import random

MAX_SCORE = 4


class RubricAssess(dspy.Signature):
    """Assess a narrative based on a rubric."""

    question = dspy.InputField(format=str)
    narrative = dspy.InputField()
    rubric = dspy.InputField()

    assessment = dspy.OutputField(
        desc="A single number from the options in the rubric. Provide only a single number with no other text."
    )


class BooleanAssess(dspy.Signature):
    """Assess a narrative with a yes/no question."""

    question = dspy.InputField(format=str)
    narrative = dspy.InputField()

    assessment = dspy.OutputField(desc="yes or no. Include only the word yes or no.")


class Grader:
    def __init__(
        self,
        llm=None,
        openai_api_key=None,
        metrics="all",
        sample_narratives=None,
        max_optimal_length=None,
    ):
        """
        Grades narratives

        Args:
            llm (LLM): LLM to use to grade accuracy, completeness, and fluency. One of llm or openai_api_key must be provided
            openai_api_key (string): OpenAI API key to use to grade accuracy, completeness, and fluency
            metrics (list of strings or "all"): One or more of  accuracy", "completeness", "fluency", "conciseness"
            sample_narratives (list of strings): Sample narratives to use to grade fluency
            max_optimal_length (int): Hyperparameter for conciseness metric, defaults to number of words in longest sample narrative or 100 if not given
        """
        self.metrics = metrics

        if metrics == "all":
            self.metrics = ["accuracy", "completeness", "fluency", "conciseness"]

        self.metric_funcs = []
        # TODO: CLEAN THIS UP TO DIRECTLY TAKE FUNCTION FROM NAME
        if "accuracy" in metrics:
            self.metric_funcs.append(accuracy)
        if "completeness" in metrics:
            self.metric_funcs.append(completeness)
        if "fluency" in metrics:
            self.metric_funcs.append("fluency")
        if "conciseness" in metrics:
            self.metric_funcs.append("conciseness")

        self.sample_narratives = sample_narratives
        self.max_optimal_length = max_optimal_length
        if max_optimal_length is None and self.sample_narratives is not None:
            self.max_optimal_length = max(
                [len(narrative.split()) for narrative in self.sample_narratives]
            )
        if self.max_optimal_length is None:
            self.max_optimal_length = 100

        self.grader_llm = llm
        self.openai_api_key = openai_api_key
        if self.grader_llm is None and self.openai_api_key is not None:
            self.grader_llm = dspy.OpenAI(
                model="gpt-4o", api_key=self.openai_api_key, max_tokens=1000
            )

    def __call__(self, explanation, explanation_format, narrative, trace=None):
        results = {}
        input_ = dspy.Example(
            explanation=explanation, explanation_format=explanation_format
        )
        output_ = dspy.Prediction(narrative=narrative)

        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy(
                input_, output_, grader=self.grader_llm, trace=trace
            )
        if "completeness" in self.metrics:
            results["completeness"] = completeness(
                input_, output_, grader=self.grader_llm, trace=trace
            )
        if "fluency" in self.metrics:
            results["fluency"] = fluency(
                input_,
                output_,
                grader=self.grader_llm,
                trace=trace,
                good_narratives=self.sample_narratives,
            )
        if "conciseness" in self.metrics:
            results["conciseness"] = conciseness(
                input_, output_, max_optimal_length_per_feature=self.max_optimal_length
            )

        if trace is None:
            return pd.Series(results)
        else:
            return (
                (results["accuracy"] == MAX_SCORE)
                and (results["fluency"] == MAX_SCORE)
                and (results["completeness"] == MAX_SCORE)
                and (results["conciseness"] >= 3.5)
            )


def compute_score_from_boolean(metric, question, narrative, grader, iters=3):
    total_score = 0.0

    with dspy.context(lm=grader):
        for i in range(iters):
            score = dspy.Predict(BooleanAssess)(
                question=question, narrative=narrative
            ).assessment.lower()
            if score == "yes":
                total_score += 1
            elif score == "no":
                pass
            else:
                print("Invalid score for metric %s: %s" % (metric, score))
    score = total_score / iters

    if 0.3 < score < 0.7:
        print("Inconsistent score for metric %s: %s" % (metric, score))

    return score * MAX_SCORE


def compute_score_from_rubric(
    metric, question, rubric, narrative, grader, iters=3, rational_type=None
):
    scores = []
    with dspy.context(lm=grader):
        for i in range(iters):
            if rational_type is None:
                score = dspy.Predict(RubricAssess)(
                    question=question, rubric=rubric, narrative=narrative
                ).assessment
            else:
                score = dspy.ChainOfThought(RubricAssess, rationale_type=rational_type)(
                    question=question,
                    rubric=rubric,
                    narrative=narrative,
                ).assessment
            try:
                scores.append(int(score))
            except ValueError:
                print("Invalid score for metric %s: %s" % (metric, score))

    if 0 in scores and MAX_SCORE in scores:
        print("Inconsistent score for metric %s: %s" % (metric, scores))

    return sum(scores) / iters


def accuracy(input_, output_, grader, trace=None):
    question = (
        f"How accurate is the information in the narrative, based on the explanation given? "
        f"A narrative can score 4 even if it is missing information as long as everything in the narrative is correct. "
        f"Make sure the contribution direction is correct - positive contributions increase the output, negative contributions decrease the output."
        f"\n\nExplanation format: {input_.explanation_format}.\nExplanation: {input_.explanation}"
    )
    rubric = f"0 - Contains one or more errors in value or contribution direction. 4 - Contains no errors, but may be missing information."

    rational_type = dspy.OutputField(
        prefix="Start by listing out all the features in the narrative, and then for each one compare it to the explanation to ensure its value and contribution are approximately correct.",
    )

    return compute_score_from_rubric(
        "accuracy",
        question,
        rubric=rubric,
        narrative=output_.narrative,
        grader=grader,
        rational_type=rational_type,
    )


def fluency(input_, output_, grader, trace=None, good_narratives=None):
    if good_narratives is None:
        question = f"How natural and human is the narrative?"
    else:
        question = f"How well does the style of the narrative match the style of the example narratives? Consider only the linguistic style, not the topic. Example narratives:"
        for narrative in good_narratives:
            question += f"\n{narrative}"
    if good_narratives is not None:
        rubric = f"0: Very dissimilar. 1: Dissimilar. 2: Neutral. 3: Similar. 4: Very similar"
    else:
        rubric = (
            f"0: Very unnatural. 1: Unnatural. 2: Neutral. 3: Natural. 4: Very natural"
        )
    return compute_score_from_rubric(
        "fluency", question, rubric, output_.narrative, grader
    )


def completeness(input_, output_, grader, trace=None):
    question = f"How completely does the narrative below describe the explanation given in <<>>?\nExplanation format: {input_.explanation_format}.\nExplanation: <<{input_.explanation}>>"
    rubric = "0 - One or more feature names from the explanation are not mentioned at all in the narrative. 2 - All features are mentioned, but not all feature values and/or contribution directions. 4 - All features are mentioned, and for each feature, includes at least an approximation of the feature's value and contribution direction."
    rational_type = dspy.OutputField(
        prefix="Start by listing out all the features in the explanations, and then determine every feature is present in the narrative, along with its value and contribution direction.",
    )

    return compute_score_from_rubric(
        "completeness",
        question,
        rubric,
        output_.narrative,
        grader,
        rational_type=rational_type,
    )


def conciseness(
    input_, output_, grader=None, trace=None, max_optimal_length_per_feature=20
):
    num_features = input_.explanation.count("(")
    if num_features == 0:
        num_features = 1
    length = len(output_.narrative.split())
    max_optimal_length = max_optimal_length_per_feature * num_features
    # scale length between 0 and 2
    return max(
        0.0,
        min(
            MAX_SCORE,
            MAX_SCORE * (2 - length / max_optimal_length),
        ),
    )


def context_awareness(input_, output_, grader, trace=None):
    question = (
        f"How well does the rationalization help explain the logic in the narrative?"
    )
    rubric = f"0: Not at all. 2: Somewhat. 4: Very well."
    narrative_input = (
        f"Narrative: {output_.narrative}. Rationalization: {output_.rationalization}"
    )
    return compute_score_from_rubric(
        "context_awareness", question, rubric, narrative_input, grader
    )
