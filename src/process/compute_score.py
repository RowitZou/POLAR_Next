import argparse
import os
import json
import re
from src.utils import SEED


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        try:
            assert s[: len(left)] == left
        except:
            return None
        return s[len(left):]

    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
    except:
        return None

    return s[len(left): -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx: right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    ans = remove_boxed(last_boxed_only_string(solution_str))
    return ans.strip() if ans is not None else None


def extract_thinking_content(text: str) -> tuple[str, str]:
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking_content = match.group(1).strip()
        remaining_content = match.group(2).strip()
        return thinking_content, remaining_content
    return "", text


def yield_data(input_path): 

    # If input_path is a directory
    if os.path.isdir(input_path):
        root_dir = input_path
        for file in os.listdir(root_dir):
            if "output" in file:
                continue
            with open(os.path.join(root_dir, file), "r", encoding="utf8") as f:
                for line in f:
                    item = json.loads(line)
                    yield item

    else:
        with open(input_path, "r", encoding="utf8") as f:
            for line in f:
                item = json.loads(line)
                yield item


def score(input_path, output_path, already_processed_lines=0):

    score_cache = dict()
    fw = open(output_path, "a", encoding="utf8")

    for n, item in enumerate(yield_data(input_path)):

        if n < already_processed_lines:
            continue

        if "<think>" not in item["output"]:
            item["output"] = "<think>\n" + item["output"]

        _, solution_str = extract_thinking_content(item["output"])
        ans = extract_solution(solution_str)
        if ans is None:
            ans = solution_str.strip()

        ground_truth = item["gts"]

        if (ground_truth, ans) in score_cache:
            score = score_cache[(ground_truth, ans)]
        else:
            score, _, _, _ = SEED(ground_truth, ans, "Expression")
            score_cache[(ground_truth, ans)] = score

        item["score"] = score
        item["reward"] = score
        fw.write(json.dumps(item, ensure_ascii=False) + "\n")
        fw.flush()

    fw.close()


def process(args):

    input_path = args.input
    output_path = args.output

    input_file_index_list = []
    for file_name in os.listdir(input_path):
        if file_name.endswith(".jsonl"):
            input_file_index_list.append(int(file_name.split(".jsonl")[0]))
    input_file_index_list = sorted(input_file_index_list)

    for file_index in input_file_index_list:
        if file_index < args.start or file_index > args.end:
            continue

        print("Processing file index: %d" % file_index)
        input_file_path = os.path.join(input_path, "%d.jsonl" % file_index)
        output_file_path = os.path.join(output_path, "%d.jsonl" % file_index)

        if os.path.exists(output_file_path):
            already_processed_lines = sum(1 for _ in open(output_file_path, "r", encoding="utf8"))
        else:
            already_processed_lines = 0
        print("Already processed lines: %d" % already_processed_lines)
        score(input_file_path, output_file_path, already_processed_lines)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs/verl_opd_policy_Qwen3-8B_Genral_OPD_reward_ZERO_ref_Qwen3-30B-A3B_data_cmphysbench_lr_1e-6/trajectory_data/validation', help='input path')
    parser.add_argument('--output', type=str, default='/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/outputs/verl_opd_policy_Qwen3-8B_Genral_OPD_reward_ZERO_ref_Qwen3-30B-A3B_data_cmphysbench_lr_1e-6/trajectory_data/score', help='output path')
    parser.add_argument('--start', type=int, default=0, help='start file index')
    parser.add_argument('--end', type=int, default=45, help='end file index')
    args = parser.parse_args()

    process(args)
