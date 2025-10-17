'''
Strip @@@_ token as a prefix from the modified files.
'''
import os
from tqdm import tqdm
import json
import re
from word2number import w2n

_int_re   = re.compile(r'^[+-]?\d+$')
_float_re = re.compile(r'^[+-]?\d+\.\d+$')
NUMERIC_RE = re.compile(r'^-?\d+(?:\.\d+)?$')
DASH_RE    = re.compile(r'[–—−]')
def convert_value(val,not_anomaly = False):            
    if isinstance(val, str):
        v = val.strip()
        # normalize any unicode dash → ascii hyphen-minus
        v = DASH_RE.sub('-', v)
        # if commas are only thousand-separators (no ", "), remove them
        if ', ' not in v:
            v = v.replace(',', '')
            # now v is like "-1234.56" or "789"
            if NUMERIC_RE.match(v):
                return float(v) if '.' in v else int(v)
            if not_anomaly:
                try:
                    return w2n.word_to_num(v)
                except ValueError:
                    pass
    return val



    

def strip_token(input_folder_path, output_folder_path):
    files = sorted(os.listdir(input_folder_path))
    json_files = [file for file in files if file.endswith('.json') and os.path.isfile(os.path.join(input_folder_path, file))]

    # Create the output folder if not present
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for file_name in tqdm(json_files, desc="Processing files", unit="file"):
        # if file_name == "15382_updated.json":
        file_path = os.path.join(input_folder_path, file_name)
        output_file = os.path.join(output_folder_path, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                stripped_data = []
                for row in data:
                    stripped_dict = {}
                    for key, value in row.items():
                        if isinstance(value, str) and value.startswith("@@@_"):
                            s = str(value[len("@@@_"):]) # Remove the @@@ prefix
                            # print(s)
                            # print(type(s))
                            if _int_re.match(s):
                                v = int(s)

                            # 2) Well-formed float?
                            elif _float_re.match(s):
                                whole, frac = s.split('.', 1)
                                # if fractional part is all zeros, treat as int
                                if set(frac) == {"0"}:
                                    v = int(whole)
                                else:
                                    # print(file_path)
                                    v = float(s)
                            elif isinstance(s, str):
                                v = convert_value(s,not_anomaly = False)

                            # 3) Not a number at all → leave as string
                            else:
                                v = s
                            # print("\n")
                            # print(v)
                            # print(type(v))
                            # print("=======================")
                            stripped_dict[key] = v
                            
                        else:
                            # print(value)
                            # print(type(value))
                            s = str(value)
                            if _int_re.match(s):
                                v = int(s)

                            # 2) Well-formed float?
                            elif _float_re.match(s):
                                whole, frac = s.split('.', 1)
                                # if fractional part is all zeros, treat as int
                                if set(frac) == {"0"}:
                                    v = int(whole)
                                else:
                                    # print(file_path)
                                    v = float(s)
                            elif isinstance(s, str):
                                v = convert_value(s,not_anomaly=True)

                            # 3) Not a number at all → leave as string
                            else:
                                v = s
                            stripped_dict[key] = v

                            # print("\n")
                            # print(v)
                            # print(type(v))
                            # print("=======================")

                    stripped_data.append(stripped_dict)
                with open(output_file, 'w', encoding='utf-8') as output_json:
                    json.dump(stripped_data, output_json, indent=4, separators=(",", ":"), ensure_ascii=False)
        
            except json.JSONDecodeError as e:
                print(f"Error reading {file_name}: {e}")

def run(input_folder_path,output_folder_path):
    # folders = [name for name in os.listdir(input_folder_path)
    #        if os.path.isdir(os.path.join(input_folder_path, name))]

    # for fold in folders:
    #     if fold != "Value_Anomaly_FetaQA":
    #         in_fold = os.path.join(input_folder_path,fold)
    #         out_fold= os.path.join(output_folder_path,fold)
    #         strip_token(in_fold, out_fold)
    in_fold = os.path.join(input_folder_path)
    out_fold= os.path.join(output_folder_path)
    strip_token(in_fold, out_fold)
if __name__ == "__main__":
    DIR = ["variation_1","variation_2","variation_3"]
    FOLDS = ["FeTaQA", "Spider_Beaver", "WikiTQ"]

    for d in DIR :
        for f in FOLDS:
            # Hardcoded input and output folder paths
            input_folder_path = f"path_to_dataset/{d}/{f}/Merged/" # Replace this with the actual input folder path
            output_folder_path = f"path_to_dataset/{d}/{f}/Merged-str"
            # strip_token(input_folder_path, output_folder_path)
            run(input_folder_path,output_folder_path)