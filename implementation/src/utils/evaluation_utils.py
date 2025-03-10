import sys
import os
import re

def rename(input: str):
    input = input.replace("sc", "(n=3)")
    input = input.replace("s", "-shot")
    input = input.replace("_", " ")
    return input

def create_run_label_exp1(input: str):
    return input

def create_run_label_exp2(input: str):
    label = rename(input.split("/")[-1])
    return label

def create_run_label_exp3(input: str):
    return input.split("/")[-1]

def extract_number(element):
    try:
        return int(element.split('-')[-1])
    except:
        return sys.maxsize
    
def extract_prefix(folderpath):
    # Extract the base name from the path
    basename = os.path.basename(folderpath)
    
    # Use a regular expression to match the prefix
    # Here, we assume the prefix is followed by "_few_shot" or "_zero_shot"
    match = re.match(r'(.+?)_(?:few_shot|zero_shot)', basename)
    
    if match:
        return match.group(1)
    else:
        print("The folder name does not match the expected pattern")
        return basename