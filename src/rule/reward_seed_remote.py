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

"""
SEED Reward Function with Remote Server Support

This module provides reward computation using SEED scores, with support for both
local computation and remote server-based computation to avoid GPU timeout issues.

Usage:
    # Use remote server (recommended for GPU training)
    from reward_seed_remote import compute_score_batch
    
    # Configure server address via environment variable
    # export SEED_SERVER_ADDRESS="127.0.0.1:30001"
"""

import re
import sys
import os
from typing import List, Optional

# Add src directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)  # src/
_project_root = os.path.dirname(_src_dir)  # POLAR_Next/
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Server configuration - can be overridden via environment variable
SEED_SERVER_ADDRESS = os.environ.get('SEED_SERVER_ADDRESS', '10.102.243.64:30030')
SEED_SERVER_TIMEOUT = float(os.environ.get('SEED_SERVER_TIMEOUT', '600.0'))

# Import client using absolute import
from utils.seed.seed_client import SEEDClient, get_seed_client


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


# Global client instance
_client: Optional[SEEDClient] = None


def get_client() -> SEEDClient:
    """Get or create the SEED client."""
    global _client
    if _client is None:
        _client = SEEDClient(
            server_address=SEED_SERVER_ADDRESS,
            timeout=SEED_SERVER_TIMEOUT
        )
    return _client


def compute_score_batch(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    prompt_key="prompt"
) -> List[float]:
    """Compute scores for a batch of data using the SEED server for VERL.

    This function uses the remote SEED server to compute scores, avoiding
    timeout issues in GPU training environments.

    Args:
        data_sources: List of data sources.
        solution_strs: List of solution strings.
        ground_truths: List of ground truth strings or {"role": xxx, "content": xxx} messages.
        extra_infos: List of extra information dictionaries containing prompt_key,
            which is the dictionary-style input prompt for policy model.

    Returns:
        scores: A list of computed scores for each data source.
    """

    # Skip test set evaluation
    if extra_infos[0]["split"] == "test":
        return [0. for _ in range(len(solution_strs))]

    # Prepare batch data
    references = []
    outputs = []
    valid_indices = []
    
    for i, (data_source, solution_str, ground_truth, extra_info) in enumerate(zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    )):
        _, solution_str = extract_thinking_content(solution_str)
        extracted_solution_str = extract_solution(solution_str)
        
        # Check validity - only check for None, length limit removed since server has timeout control
        if extracted_solution_str is None:
            outputs.append(None)  # Mark as invalid
        else:
            outputs.append(extracted_solution_str)
        
        references.append(ground_truth)
        valid_indices.append(i)

    # Separate valid and invalid
    valid_refs = []
    valid_outs = []
    valid_idx_map = []  # Maps position in valid list to original index
    
    scores = [0.0] * len(solution_strs)  # Initialize all scores to 0
    
    for i, (ref, out) in enumerate(zip(references, outputs)):
        if out is not None:
            valid_refs.append(ref)
            valid_outs.append(out)
            valid_idx_map.append(i)
    
    # If no valid outputs, return all zeros
    if not valid_refs:
        return scores
    
    # Call SEED server for valid outputs
    client = get_client()
    result = client.compute_batch(
        references=valid_refs,
        outputs=valid_outs,
        types=["Expression"] * len(valid_refs)
    )
    
    if result is None:
        # Server request failed, return zeros
        return scores
    
    # Map scores back to original indices
    server_scores = result.get('scores', [0.0] * len(valid_refs))
    for idx, orig_idx in enumerate(valid_idx_map):
        if idx < len(server_scores):
            scores[orig_idx] = server_scores[idx]
    
    return scores


def compute_score_batch_local_fallback(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    prompt_key="prompt"
) -> List[float]:
    """
    Compute scores with local fallback if server is unavailable.
    
    First tries the remote server, falls back to local computation if server fails.
    """
    # Try remote first
    client = get_client()
    if client.health_check():
        return compute_score_batch(
            data_sources, solution_strs, ground_truths, extra_infos, prompt_key
        )
    
    # Fall back to local computation
    print("SEED server unavailable, falling back to local computation")
    from utils import SEED
    
    if extra_infos[0]["split"] == "test":
        return [0. for _ in range(len(solution_strs))]

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
