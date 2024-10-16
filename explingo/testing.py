import dspy


class MockNarratorLLM:
    def __init__(self, response, **kwargs):
        """
        Create a mock LLM for testing purposes

        Args:
            response (String): Narrative response expected from the LLM
        """
        self.response = response
        self.kwargs = kwargs
        self.history = []

    def __call__(self, prompt=None, **kwargs):
        completions = "Narrative: " + self.response
        self.history.append({"prompt": prompt, "completions": completions})
        return [completions]

    def copy(self, **kwargs):
        return self.__class__(self.response, **kwargs)

    def inspect_history(self):
        print(self.history)


class MockGraderLLM:
    def __init__(self, response, **kwargs):
        """
        Create a mock Grader for testing purposes

        Args:
            response (int): Grader response expected from the Grader
        """
        self.response = response
        self.kwargs = kwargs
        self.history = []

    def __call__(self, prompt=None, *args, **kwargs):
        completions = "Assessment: " + str(self.response)
        self.history.append({"prompt": prompt, "completions": completions})
        return [completions]

    def copy(self, **kwargs):
        return self.__class__(self.response, **kwargs)

    def inspect_history(self):
        print(self.history)
