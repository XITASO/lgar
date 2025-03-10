import os
import json
import datetime
from typing import Any, Dict

def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists. Create it if it does not exist.

    :param path: Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_json(data: Dict, filename: str, output_dir: str) -> None:
    """
    Save data to a JSON file.

    :param data: Data to save.
    :param filename: Filename for the saved data.
    :param output_dir: Output directory.
    """
    if not os.path.exists(output_dir):
        ensure_directory_exists(output_dir) 
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_json_file(file_path: str) -> Any:
    """
    Loads a JSON file and returns its content.

    :param file_path: Path to the JSON file.
    :return: Content of the JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from {file_path}: {e}")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to read file {file_path}: {e}")


def generate_foldername(
    name_of_model: str,
    length: int,
    slr_tag: str,
    is_few_shot: bool,
    index: int,
) -> str:
    """
    Generate a foldername string based on model information, SLR tag, and current timestamp.

    :param name_of_model: Name of the model.
    :param length: Number of entries.
    :param slr_tag: Tag of the SLR.
    :param is_few_shot: Boolean, deciding if few shot prompting was used or not.
    :param index: Index of the experiment.
    :return: Generated foldername
    """
    current_time = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    few_shot = "few_shot" if is_few_shot else "zero_shot"
    return f"{slr_tag}_{few_shot}_{name_of_model}_{current_time}_{length}_{index}/"

def scan_folder_for_csv(folder_path):
    file_list = os.listdir(folder_path)
    
    csv_files = [folder_path + file for file in file_list if file.endswith('.csv')]
    
    return csv_files

def save_ranked_df_with_reranker(document_ids, reranker: str, path: str, sm: str, duration: float):
    query = "title + rqs"
    if sm == "system_message_basic":
        query = "title"
    data_to_save = {
        "reranker": reranker,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "duration": duration,
        "ids": document_ids
    }
    with open(path, "w") as outfile:
        json.dump(data_to_save, outfile, indent=4)
