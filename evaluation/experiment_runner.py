import metrics
import os
import examples
import random
from explingo import Explingo


class ExplingoExperimentRunner:
    def __init__(
        self, llm, dataset_filepath, openai_api_key, verbose=0, save_results=True
    ):
        (
            self.labeled_train,
            self.labeled_eval,
            self.unlabeled_train,
            self.unlabeled_eval,
        ) = examples.get_data(dataset_filepath)
        self.train_data = self.labeled_train + self.unlabeled_train
        self.eval_data = self.labeled_eval + self.unlabeled_eval
        assert len(self.train_data) == 10
        print(dataset_filepath)
        print(f"Total number of examples: {len(self.train_data) + len(self.eval_data)}")
        print(f"Labeled training examples: {len(self.labeled_train)}")
        print(f"Labeled evaluation examples: {len(self.labeled_eval)}")
        print(f"Unlabeled training examples: {len(self.unlabeled_train)}")
        print(f"Unlabeled evaluation examples: {len(self.unlabeled_eval)}")

        max_optimal_length = max(
            [
                len(d.narrative.split()) / d.explanation.count("(")
                for d in self.labeled_train
            ]
        )
        print("Max optimal length:", max_optimal_length)
        print("---")

        example_good_narratives = random.sample(
            [d.narrative for d in self.labeled_train], 5
        )

        self.metrics = metrics.Metrics(
            metric_funcs=[
                metrics.accuracy,
                metrics.completeness,
                metrics.fluency,
                metrics.conciseness,
            ],
            openai_key=openai_api_key,
            verbose=verbose,
            metric_kwargs={
                "conciseness": {"max_optimal_length_per_feature": max_optimal_length},
                "fluency": {"good_narratives": example_good_narratives},
            },
        )

        self.verbose = verbose
        self.save_results = save_results

        self.explingo = Explingo(
            llm,
            context=self.labeled_train[0]["context"],
            labeled_train_data=self.labeled_train,
            unlabeled_train_data=self.unlabeled_train,
        )

    def run_experiment(self, func, prompt=None, max_iters=100, kwargs=None):
        if kwargs is None:
            kwargs = {}

        total_scores = None
        results = []
        for i, example in enumerate(self.eval_data):
            if i >= max_iters:
                break
            result = func(
                prompt=prompt,
                explanation=example.explanation,
                explanation_format=example.explanation_format,
                **kwargs,
            )
            if result is not None:
                score = self.metrics(example, result)
                if total_scores is None:
                    total_scores = score[1]
                else:
                    total_scores += score[1]
                if self.verbose >= 1:
                    print("Explanation:", example.explanation)
                    print("Narrative:", result.narrative)
                    print("Total Score:", score[0])
                    print(
                        "".join(
                            f"{metric}: {score}, " for metric, score in score[1].items()
                        )
                    )
                    print("--")
                if self.save_results:
                    results.append(
                        {
                            "func": func.__name__,
                            "prompt": kwargs.get("prompt", ""),
                            "n_few_shot": kwargs.get("n_few_shot", 0),
                            "n_labeled_few_shot": kwargs.get("n_labeled_few_shot", 0),
                            "n_bootstrapped_few_shot": kwargs.get(
                                "n_bootstrapped_few_shot", 0
                            ),
                            "explanation": example.explanation,
                            "narrative": result.narrative,
                            "scores": "".join(
                                f"{metric}: {score}, "
                                for metric, score in score[1].items()
                            ),
                        }
                    )

        total = min(max_iters, len(self.eval_data))
        average_scores = total_scores / total
        total_average_score = total_scores.sum() / total

        if self.save_results:
            return total_average_score, average_scores, results
        return total_average_score, average_scores

    def run_basic_prompting_experiment(self, prompt=None, max_iters=100):
        """
        Run a basic prompting experiment

        Args:
            prompt (string): Prompt
            max_iters (int): Maximum number of examples to run on

        Returns:
            total_average_score (float): Average total score over all explanations
            average_scores (pd.Series): Average scores for each metric
        """
        return self.run_experiment(
            self.explingo.basic_prompt,
            prompt=prompt,
            max_iters=max_iters,
        )

    def run_few_shot_experiment(self, prompt=None, max_iters=100, n_few_shot=3):
        """
        Run a few-shot experiment

        Args:
            prompt (string): Prompt
            max_iters (int): Maximum number of examples to run on
            n_few_shot (int): Number of examples to use in few-shot learning

        Returns:
            total_average_score (float): Average total score over all explanations
            average_scores (pd.Series): Average scores for each metric
        """
        return self.run_experiment(
            self.explingo.few_shot,
            prompt=prompt,
            max_iters=max_iters,
            kwargs={"n_few_shot": n_few_shot},
        )

    def run_bootstrap_few_shot_experiment(
        self,
        prompt=None,
        max_iters=100,
        n_labeled_few_shot=3,
        n_bootstrapped_few_shot=3,
    ):
        """
        Run a bootstrap few-shot experiment
        Args:
            prompt (string): Prompt
            max_iters (int): Maximum number of examples to run on
            n_labeled_few_shot (int): Number of examples to use in few-shot learning
            n_bootstrapped_few_shot (int): Number of  bootstrapped examples to use in few-shot learning

        Returns:
            total_average_score (float): Average total score over all explanations
            average_scores (pd.Series): Average scores for each metric
        """
        return self.run_experiment(
            self.explingo.bootstrap_few_shot,
            prompt=prompt,
            max_iters=max_iters,
            kwargs={
                "metric": self.metrics,
                "n_labeled_few_shot": n_labeled_few_shot,
                "n_bootstrapped_few_shot": n_bootstrapped_few_shot,
            },
        )
