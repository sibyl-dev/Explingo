import explingo


def test_grader_run_metrics():
    response = 4
    mock_grader_llm = explingo.testing.MockGraderLLM(response)
    grader = explingo.Grader(llm=mock_grader_llm, metrics="all")
    result = grader("explanation", "explanation_format", "narrative")
    for metric in ["accuracy", "fluency", "conciseness", "completeness"]:
        assert result[metric] == response
