# <img src="https://github.com/sibyl-dev/Explingo/blob/main/parrot.jpg" width="auto" height="75"> Explingo

# Explingo
Transform your ML explanations into human-friendly natural-language narratives.

NOTE: Explingo is still under active development and currently only supports a few basic explanation types
and GPT-API models. 

## Installation
Explingo can be installed through PIP
```bash
pip install explingo
```

## Usage
To transform explanations into narratives, you can use the Narrator class.
```python
from explingo import Narrator, Grader 

example_narratives = [
    ("(Above ground living area square feet, 1256.00, -12527.46), (Overall material and finish of the house, 5.00, -10743.76), (Second floor square feet, 0.00, -10142.29)", 
     "The house's living area size of around 1,200 sq. ft., lower quality materials (5/10), and lack of a second floor are the main reasons for the low price."),
    ("(Second floor square feet, 854.00, 12757.84), (Original construction date, 2003.00, 9115.72)",
     "The house's large second floor of around 850 sq. ft and recent construction date of 2003 increases its value."),
    ("(Overall material and finish of the house, 8.00, 10743.76), (Above ground living area square feet, 2000.00, 12527.46), (Second floor square feet, 1000.00, 10142.29)",
        "The house's high quality materials (8/10), large living area size of around 2,000 sq. ft., and a second floor of around 1,000 sq. ft. are the main reasons for the high price."),
]

explanation_format = "(feature name, feature value, SHAP feature contribution)"
context = "The model predicts house prices"

narrator = Narrator(openai_api_key=[OPENAI_API_KEY], 
                    explanation_format=explanation_format,
                    context=context,
                    labeled_train_data=example_narratives)

explanation = "(number of bathrooms, 3, 7020), (number of bedrooms, 4, 12903)"

narrative = narrator.narrate(explanation)
```

To evaluate the quality of the generated narratives, you can use the Grader class.
```python
grader = Grader(openai_api_key=[OPENAI_API_KEY], 
                metrics="all", 
                sample_narratives=[narrative[1] for narrative in example_narratives])

metrics = grader(explanation=explanation, 
                 explanation_format=explanation_format, 
                 narrative=narrative)
```
