zero_shot_prompt_template = """
Task:
You will be presented with a paper's title and abstract. Your task is to decide how relevant the given paper is to the review. Return a number for you decision ranging from '{relevance_lower_value}' to '{relevance_upper_value}', where '{relevance_lower_value}' means that you are absolutely sure that the paper should be excluded, where '{relevance_upper_value}' means that you are absolutely sure that the paper should be included, and where an intermediate value means that you are unsure. Please read the title and the abstract carefully and then make your decision based on the provided inclusion and exclusion criteria.

Title: '{title_paper}'
Abstract: '{abstract}'

Inclusion criteria: '{inclusion_criteria}'
Exclusion criteria: '{exclusion_criteria}'

Give your answer in the following format:
```
Decision: {relevance_lower_value} - {relevance_upper_value}
```
"""

zero_shot_prompt_template_binary = """
Task:
You will be presented with a paper's title and abstract. Your task is to decide if the given paper is relevant to the review. Use the inclusion and exclusion criteria provided below to inform your decision. If any exclusion criteria are met or not all inclusion criteria are met, the paper should be excluded. If all inclusion criteria are met and no exclusion criterion is met, the paper should be included. Return '{relevance_lower_value}' if the paper should be excluded and '{relevance_upper_value}' if the paper should be included. Please read the title and the abstract carefully and then make your decision based on the provided criteria.

Title: '{title_paper}'
Abstract: '{abstract}'

Inclusion criteria: '{inclusion_criteria}'
Exclusion criteria: '{exclusion_criteria}'

Give your answer in the following format:
```
Decision: {relevance_lower_value} or {relevance_upper_value}
```
"""

CoT_prompt_template = """
Task:
You will be presented with a paper's title and abstract. Your task is to decide how relevant the given paper is to the review. Return a number for you decision ranging from '{relevance_lower_value}' to '{relevance_upper_value}', where '{relevance_lower_value}' means that you are absolutely sure that the paper should be excluded, where '{relevance_upper_value}' means that you are absolutely sure that the paper should be included, and where an intermediate value means that you are unsure. Please read the title and the abstract carefully and then make your decision based on the provided inclusion and exclusion criteria. Think step by step.

Title: '{title_paper}'
Abstract: '{abstract}'

Inclusion criteria: '{inclusion_criteria}'
Exclusion criteria: '{exclusion_criteria}'

Give your answer in the following format:
```
Explanation: "Let's think step by step..."
---
Decision: {relevance_lower_value} - {relevance_upper_value}
```

Explanation:
"""