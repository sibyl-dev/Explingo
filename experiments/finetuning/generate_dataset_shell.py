import pandas as pd
import os
import json


def parse_exp(exp, num_features=5, include_average=True):
    features = []
    if num_features is None:
        num_features = len(exp)
    for i in range(num_features):
        if include_average:
            features.append(
                "({}, {}, {}, {})".format(
                    exp[i]["Feature Name"].strip(),
                    exp[i]["Feature Value"],
                    exp[i]["Contribution"],
                    exp[i]["Average/Mode"],
                )
            )
        else:
            features.append(
                "({}, {}, {})".format(
                    exp[i]["Feature Name"].strip(),
                    exp[i]["Feature Value"].strip(),
                    exp[i]["Contribution"],
                )
            )
    return ", ".join(features)


prompts = [
    "You are helping users understand an ML model's predictions. "
    "I will give you feature contribution explanations, generated using SHAP, in "
    "(feature, feature_value, contribution, average_feature_value) format. Convert the "
    "explanations in simple narratives. Do not use more tokens than necessary.",
]

exp_files = [
    os.path.join("..", "data", "ames_housing_{}.csv".format(i)) for i in range(10)
]
exps = []
for exp in exp_files:
    exp_df = pd.read_csv(exp)
    exp_df = exp_df.sort_values(by="Contribution", key=abs, ascending=False)
    exps.append(exp_df.to_dict("records"))

save_json = []
for prompt in prompts:
    for exp in exps:
        save_json.append(
            {
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": parse_exp(exp)},
                    {"role": "assistant", "content": ""},
                ]
            }
        )

with open("finetuning_data_shell.json", "w") as fp:
    json.dump(save_json, fp)
