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


class Metrics:
    def __init__(self, metric_funcs, openai_key, verbose=0, metric_kwargs=None):
        self.metric_funcs = metric_funcs
        self.verbose = verbose
        self.metric_kwargs = metric_kwargs if metric_kwargs is not None else {}
        self.grader = dspy.OpenAI(
            model="gpt-4-1106-preview",
            max_tokens=1000,
            model_type="chat",
            api_key=openai_key,
        )

    def __call__(self, input_, output_, trace=None):
        metrics = {}
        for metric in self.metric_funcs:
            metric_name = metric.__name__
            kwargs = self.metric_kwargs.get(metric_name, {})
            metrics[metric_name] = metric(
                input_, output_, grader=self.grader, trace=trace, **kwargs
            )

        total_score = sum(metrics.values())

        if self.verbose == 2:
            print("Explanation:", input_.explanation)
            print("Narrative:", output_.narrative)
            print("Rationalization:", output_.rationalization)
            print("Total Score:", total_score)
            print("".join(f"{metric}: {score}, " for metric, score in metrics.items()))
            print("--")
        if self.verbose == 1:
            print("Narrative:", output_.narrative)
            print("Total Score:", total_score)
            print("".join(f"{metric}: {score}, " for metric, score in metrics.items()))
            print("--")

        if trace is None:
            return total_score, pd.Series(metrics)
        else:
            return (metrics["accuracy"] == MAX_SCORE) and (
                total_score >= len(metrics) * MAX_SCORE
            )


def compute_score_from_boolean(metric, question, narrative, grader, iters=5):
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
    metric, question, rubric, narrative, grader, iters=5, rational_type=None
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
            scores.append(int(score))

    if 0 in scores and MAX_SCORE in scores:
        print("Inconsistent score for metric %s: %s" % (metric, scores))

    return sum(scores) / iters


def accuracy(input_, output_, grader, trace=None):
    question = (
        f"How accurate is the information in the narrative, based on the explanation given? "
        f"A narrative can score 4 even if it is missing information as long as everything in the narrative is correct. "
        f"\n\nExplanation format: {input_.explanation_format}.\nExplanation: {input_.explanation}"
    )
    rubric = f"0 - Contains one or more errors in value or contribution direction. 4 - Contains no errors."
    return compute_score_from_rubric(
        "accuracy", question, rubric=rubric, narrative=output_.narrative, grader=grader
    )


def fluency(
    input_, output_, grader, trace=None, good_narratives=None, bad_narratives=None
):
    if good_narratives is None:
        question = f"How natural and human is the narrative?"
    else:
        question = f"How well does the style of the narrative match the style of these examples:"
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
        desc="Feature-by-feature processing of the narrative.",
    )

    return compute_score_from_rubric(
        "completeness",
        question,
        rubric,
        output_.narrative,
        grader,
        rational_type=rational_type,
    )


def conciseness(input_, output_, grader=None, trace=None, max_optimal_length=100):
    length = len(output_.narrative.split())
    # scale length between 0 and 2
    return max(0.0, min(MAX_SCORE, MAX_SCORE * (2 - length / max_optimal_length)))


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
