# Copyright 2025 POLAR Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__name__), '..', 'src'))
from utils import SEED


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

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


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, prompt_key="prompt"):
    """Compute scores for a batch of data using the POLAR reward model for VERL.

    Args:
        data_sources: List of data sources.
        solution_strs: List of solution strings.
        ground_truths: List of ground truth strings or {"role": xxx, "content": xxx} messages.
        extra_infos: List of extra information dictionaries containing prompt_key,
            which is the  dictionary-style input prompt for policy model and POLAR.

    Returns:
        scores: A list of computed scores for each data source.
    """

    batch_data = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):

        _, solution_str = extract_thinking_content(solution_str)
        extracted_solution_str = extract_solution(solution_str)

        data = {
            "prompt": extra_info[prompt_key],
            "reference": ground_truth,
            "output": extracted_solution_str,
            "wrapper": "sft"
        }
        batch_data.append(data)

    scores = []

    for item in batch_data:
        if item["output"] is None or len(item["output"].strip()) > 1000:
            scores.append(0.)
            continue
        score, _, _, _ = SEED(item["reference"], item["output"], "Expression")
        scores.append(score)

    return scores
