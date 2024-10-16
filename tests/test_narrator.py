import explingo


def test_narrator():
    narrator = explingo.Narrator(explanation_format="test", context="test")
    assert narrator is not None


def test_narrate_basic_prompt():
    response = "narrative"
    mock_llm = explingo.testing.MockNarratorLLM(response)
    narrator = explingo.Narrator(
        llm=mock_llm, explanation_format="test", context="test"
    )
    explanation = "explanation"
    assert narrator.narrate(explanation) == response


def test_narrative_few_shot():
    response = "narrative"
    mock_llm = explingo.testing.MockNarratorLLM(response)
    narrator = explingo.Narrator(
        llm=mock_llm,
        explanation_format="test",
        context="test",
        sample_narratives=["sample 1", "sample 2"],
    )
    explanation = "explanation"
    assert narrator.narrate(explanation, n_examples=2) == response


def test_narrative_bootstrapped_few_shot():
    response = "narrative"
    mock_llm = explingo.testing.MockNarratorLLM(response)
    mock_grader = explingo.Grader(
        llm=explingo.testing.MockGraderLLM(4),
        metrics=["fluency, conciseness"],
        sample_narratives=["sample 1", "sample 2"],
    )
    narrator = explingo.Narrator(
        llm=mock_llm,
        explanation_format="test",
        context="test",
        sample_narratives=["sample 1", "sample 2"],
    )
    explanation = "explanation"
    narrator.narrate(explanation, n_examples=2, n_bootstrapped=2, grader=mock_grader)
