from dspy import Signature, InputField, OutputField
import dspy

GPT = dspy.OpenAI(model="gpt-4-1106-preview", max_tokens=1000, model_type="chat")


class RubricAssess(Signature):
    """Assess a narrative based on a rubric."""

    narrative = InputField()
    question = InputField()
    rubric = InputField()

    assessment_score = OutputField(
        "0, 1, or 2, based on the rubric. Only include the number."
    )


class BooleanAssess(Signature):
    """Assess a narrative with a yes/no question."""

    narrative = InputField()
    question = InputField()

    assessment_score = OutputField("yes or no")


def compute_score_from_boolean(metric, question, narrative_explanation, iters=10):
    total_score = 0

    with dspy.context(lm=GPT):
        for i in range(iters):
            score = dspy.Predict(BooleanAssess)(
                question=question, narrative_explanation=narrative_explanation
            )
            if score == "yes":
                total_score += 1
    score = total_score / iters

    if 0.3 < score < 0.7:
        print("Inconsistent score for metric %s: %s" % (metric, score))

    return score * 2


def compute_score_from_rubric(metric, question, rubric, narrative_explanation, iters=5):
    scores = []

    with dspy.context(lm=GPT):
        for i in range(iters):
            score = dspy.Predict(RubricAssess)(
                question=question,
                rubric=rubric,
                narrative_explanation=narrative_explanation,
            )
            scores.append(score)

    if 0 in scores and 1 in scores:
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
    return 2 - min(length / 50, 2)
