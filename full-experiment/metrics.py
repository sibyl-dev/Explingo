import dspy
import pandas as pd

grader = dspy.OpenAI(model="gpt-4-1106-preview", max_tokens=1000, model_type="chat")


class RubricAssess(dspy.Signature):
    """Assess a narrative based on a rubric."""

    narrative = dspy.InputField()
    question = dspy.InputField()
    rubric = dspy.InputField()

    assessment = dspy.OutputField(
        desc="0, 1, or 2, based on the rubric. Include only the number."
    )


class BooleanAssess(dspy.Signature):
    """Assess a narrative with a yes/no question."""

    narrative = dspy.InputField()
    question = dspy.InputField()

    assessment = dspy.OutputField(desc="yes or no")


class Metrics:
    def __init__(self, metric_funcs, verbose=0):
        self.metric_funcs = metric_funcs
        self.verbose = verbose

    def __call__(self, gold, pred, trace=None):
        metrics = {}
        for metric in self.metric_funcs:
            metrics[metric.__name__] = metric(gold, pred, trace)

        total_score = sum(metrics.values())

        if self.verbose == 2:
            print("Explanation:", gold.explanation)
            print("Narrative:", pred.narrative)
            print("Rationalization:", pred.rationalization)
            print("Total Score:", total_score)
            print("".join(f"{metric}: {score}, " for metric, score in metrics.items()))
            print("--")
        if self.verbose == 1:
            print("Narrative:", pred.narrative)
            print("Total Score:", total_score)
            print("".join(f"{metric}: {score}, " for metric, score in metrics.items()))
            print("--")

        if trace is None:
            return total_score, pd.Series(metrics)
        else:
            return (metrics["accuracy"] == 2) and (total_score >= 8)


def compute_score_from_boolean(metric, question, narrative, iters=10):
    total_score = 0

    with dspy.context(lm=grader):
        for i in range(iters):
            score = dspy.Predict(BooleanAssess)(
                question=question, narrative=narrative
            ).assessment
            if score == "yes":
                total_score += 1
    score = total_score / iters

    if 0.3 < score < 0.7:
        print("Inconsistent score for metric %s: %s" % (metric, score))

    return score * 2


def compute_score_from_rubric(metric, question, rubric, narrative, iters=5):
    scores = []

    with dspy.context(lm=grader):
        for i in range(iters):
            score = dspy.Predict(RubricAssess)(
                question=question,
                rubric=rubric,
                narrative=narrative,
            ).assessment
            scores.append(int(score))

    if 0 in scores and 2 in scores:
        print("Inconsistent score for metric %s: %s" % (metric, scores))

    return sum(scores) / iters


def accuracy(gold, pred, trace=None):
    question = f"How accurately does the narrative describe this explanation: {gold.explanation}?. The explanation is formatted at: {gold.explanation_format}"
    rubric = f"0: Contain an error. 1: Accurate, but misleading. 2: Accurate and clear."
    return compute_score_from_rubric("accuracy", question, rubric, pred.narrative)


def fluency(gold, pred, trace=None):
    question = f"How natural and human does the narrative sound?"
    rubric = f"0: Not at all natural. 1: Somewhat natural. 2: Natural."
    return compute_score_from_rubric("fluency", question, rubric, pred.narrative)


def completeness(gold, pred, trace=None):
    question = f"Does the narrative contain all the feature values from this explanation? {gold.explanation}? The explanation is formatted at: {gold.explanation_format}"
    return compute_score_from_boolean("completeness", question, pred.narrative)


def conciseness(gold, pred, trace=None):
    length = len(pred.narrative.split())
    # scale length between 0 and 2, such that longer lengths score lower
    return 2 - min(length / 100, 2)


def context_awareness(gold, pred, trace=None):
    question = (
        f"How well does the narrative rationalization help explain the model's logic?"
    )
    rubric = f"0: Not at all. 1: Somewhat. 2: Very well."
    return compute_score_from_rubric(
        "context_awareness", question, rubric, pred.rationalization
    )
