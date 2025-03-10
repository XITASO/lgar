# LGAR: Zero-Shot LLM-Guided Neural Ranking for Abstract Screening in Systematic Literature Reviews
This repo contains the code used for the paper [TODO: link]

## Table of Contents
- [Requirements](#requirements)
- [Required Data](#required-data)
- [First Stage of Ranking (LLM Ranker)](#first-stage-of-ranking-llm-ranker)
- [Re-Ranking results of LLMs](#re-ranking-results-of-llms)
- [Evaluation](#evaluation)
- [Dense Ranker Only](#dense-ranker-only)


## Requirements
Install the requirements listed in requirements.txt, e.g., using a conda environment or venv with ```Python=3.12```

## Required Data
- The annotated criteria and research questions are located in a info.json of the respective datafolder of the dataset, e.g., ./data/synergy/info.json
- The info.json files contain our annotations and other meta data for all SLRs of the respective dataset
- The .csv files for the SLRs can be obtained from the following sources:
    - SYNERGY: https://github.com/asreview/synergy-dataset
    - [TAR2019](https://pure.strath.ac.uk/ws/portalfiles/portal/96496914/Kanoulas_etal_CEUR_2019_CLEF_2019_technology_assisted_reviews_in_empirical_medicine_overview.pdf): https://github.com/CLEF-TAR/tar/tree/master/2019-TAR/Task2
    - [Guo et al. (2023)](https://www.jmir.org/2024/1/e48996/): https://data.mendeley.com/datasets/np79tmhkh5/1
- For running experiments on a specific dataset, it is necessary to set ```folder_path_slrs``` and ```file_path_slr_infos``` accordingly in the config.json file.

The directory structure of the data folder should look like this:
```
data
├── synergy
│   └── info.json
│   └── csv files of dataset
├── tar2019
│   ├── dta
│   │   └── info.json
│   │   └── csv files of dataset
│   ├── intervention
│   │   └── info.json
│   │   └── csv files of dataset
│   ├── prognosis
│   │   └── info.json
│   │   └── csv files of dataset
│   ├──qualitative
│   │   └── info.json
│   │   └── csv files of dataset
```

## First Stage of Ranking (LLM Ranker)
The scripts to reproduce the results of this paper are located in ./implementation/src/scripts/:
- Experiment with different scales: ```run_scale_experiment.py```
- Experiment with different Prompts: ```run_prompt_experiment.py```
- Experiment with different LLMs: ```run_llm_epxeriment.py```
- Experiment with LLM (Title only): ```run_llm_ti_experiment.py```
The scripts can be started using a batch file (e.g., when using SLURM on a cluster). All scripts except for that used for LLM with title only expect to receive the index of the array job, since the scripts run only part of the code for a given index. In all files, you need to include the paths to the directory, where the model is stored or the huggingface tag of the model.

Example usage for ```run_prompt_experiment.py```:
- ```generate_examples=True```: The script creates few-shot examples for zero-shot (```index=0```) and CoT prompting  (```index=1```) for all SLRs of the current dataset (current dataset is selected by setting the respective paths in the config.json file) at the specified location (config.folder_path_few_shot_examples).
- ```generate_examples=False```: Runs the experiment for all prompting techniques. The provided index specifies the prompting technique (```index = 0```: 2-shot, ```index = 1```: CoT, ```index = 2```: CoT (n=3), ```index = 3```: 2-shot CoT, ```index = 4```: 2-shot CoT (n=3)).
    - The script saves the results to ```config.llm_client_output_directory_path``` with the following order structure:
    ```
    ├── Prompting Technique 1
    │   ├── Dataset 1
    │   │   ├── Results of SLR 1
    │   │   │   └── log_file_0.json
    │   │   ├── Results of SLR 2
    │   │   │   └── log_file_0.json
    │   │   ...
    │   └── Dataset 2
    │   ...
    ├── Prompting Technique 2
    ...
    ```
    - The results of the LLM ranker are stored in the folder of the respective SLR, in a json log-file with an index corresponding to the run (i.e., for not self-consistency runs: log_file_0.json).

## Re-Ranking results of LLMs
This section describes how to re-rank the papers with a secondary (dense) ranker, after having completed the first stage of our ranking pipeline. There are two options for performing this step, either by using the script ```evaluate_experiments_single.py``` or by using ```evaluate_experiments.py```. Both scripts expect that ```config.folder_path_slrs``` and ```config.file_path_slr_infos``` are correctly set to the dataset that should be evaluated.

Option 1: ```evaluate_experiments_single.py```
- Evaluates all experiments that are located in the subfolders of the experiment that is to be evaluated.
- The experiment can be specified by setting ```config.llm_client_output_directory_path``` to the desired folder.
- The specified folder is expected to be a folder above the folders of different SLRs of a dataset, so e.g., ```"./implementation/data/paper/prompts/synergy/CoT/"```
- To not always manually adapt the experiment path for all experiments in a SLR dataset folder, it is possible provide the path to the parent folder, e.g., ```"./implementation/data/paper/prompts/synergy/"``` and uncomment the respective lines, which split up the folders by the index provided to the main function.

Option 2: ```evaluate_experiments.py```
- Expects that ```config.llm_client_output_directory_path``` is at the level of a dataset folder, e.g., ```"./implementation/data/paper/prompts/synergy/"```
- To use this script for re-ranking the papers: ```is_lm_only=False``` and ```rerank=True```
- The script reranks (sequentially) the papers for all SLRs of all different configurations stored in ```config.llm_client_output_directory_path```

If you want to use a different re-ranker, you need to change ```config.llm_client_config.path_to_reranker``` to a different dense ranker (if none is provided, BM25 will be used as fallback). To distinguish between different re-ranking queries, you need to set ```config.llm_client_config.system_message_type``` accordingly. If you specifiy it to be ```system_message_basic```, only the title of the respective SLR is used as query, while if you specifiy it to be ```system_message_rq```, title and rqs are used as query.


## Evaluation
This section describes how the experiments can be evaluated using the script ```evaluate_experiments.py```.

- To generate only the result csv file and not re-rank the documents again: ```rerank=False``` &#8594; the script will then directly access the already ranked list stored in ranked_df.json for each SLR
- Depending on what experiment you want to evaluate you can provide a function for ```name_label``` that renames the experiments abbreviation accordingly (either create_run_label_exp1 (scale experiment), create_run_label_exp2 (prompt experiment), or create_run_label_exp3 (all other experiments))
- For the scale experiments: ```sort=True``` &#8594; ensures an ascending order or the scales; for other experiments it should be set to false
- To evaluate only the performance of a dense ranker (without LLM): ```is_lm_only=True```
- As already described above, the script expects that you have set ```config.folder_path_slrs``` and ```config.file_path_slr_infos``` correctly to the dataset you want to evaluate. Furthermore, you need to specify, which results should be evaluated. This can be achieved by setting ```config.llm_client_output_directory_path``` to the respective results folder.

## Dense Ranker Only
- To evaluate only the performance of a dense ranker, you can use the script ```/src/scripts/run_lm_experiment.py```
- The model paths need to be set according to the models' location.
- If the script is provided with an index which is not equal to 1, the query for the ranker is the title of the SLR; otherwise the query is title and research questions of the SLR
- The evaluation of the dense ranker can be performed as described in the previous section, but it is necessary to set ```is_lm_only=True```
