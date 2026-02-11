import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

categories = [
    "GPQA_diamond",
    "mmlu_pro",
    "FS-selfies",
    "MC-selfies",
    "MG-selfies",
    "PP-selfies",
    "RP-selfies",
    "RS-selfies",
]


def check_thinking_format(prediction: str) -> bool:
    """
    检查prediction字段是否符合thinking格式要求：
    1. 字符串必须以<think>开头
    2. <think>必须在</think>之前出现
    
    Args:
        prediction: prediction字段的内容
        
    Returns:
        bool: 如果符合格式返回True，否则返回False
    """
    if not prediction or not isinstance(prediction, str):
        return False

    if not prediction.startswith("<think>") and not prediction.startswith("\n"):
        return False

    think_end_pos = prediction.find("</think>")

    # 如果没有找到</think>，也认为格式不正确
    if think_end_pos == -1:
        return False

    return True


def get_category_for_file(filename: str, categories: List[str]) -> str:
    """
    根据文件名判断属于哪个类别
    
    Args:
        filename: 文件名
        categories: 类别列表
        
    Returns:
        str: 匹配的类别名，如果没有匹配则返回None
    """
    for category in categories:
        if filename.startswith(category):
            return category
    return None


def analyze_thinking_ratio(prediction_dir: str) -> Dict[str, Dict]:
    """
    分析指定目录下所有评测文件中prediction字段的thinking格式通过率
    
    Args:
        prediction_dir: 包含评测文件的目录路径
        
    Returns:
        Dict: 每个类别的统计结果
    """
    # 存储每个类别的统计信息
    category_stats = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    
    # 遍历目录中的所有文件
    if not os.path.exists(prediction_dir):
        print(f"错误: 目录 {prediction_dir} 不存在")
        return {}
    
    for filename in os.listdir(prediction_dir):
        if not filename.endswith(".json"):
            continue
        
        # 确定文件所属类别
        category = get_category_for_file(filename, categories)
        if category is None:
            continue
        
        filepath = os.path.join(prediction_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取文件 {filepath}: {e}")
            continue
        
        # 遍历文件中的所有条目
        for key, item in data.items():
            if isinstance(item, dict) and "prediction" in item:
                prediction = item["prediction"]
                category_stats[category]["total"] += 1
                
                if check_thinking_format(prediction):
                    category_stats[category]["passed"] += 1
                else:
                    category_stats[category]["failed"] += 1
    
    return dict(category_stats)


def print_report(stats: Dict[str, Dict], prediction_dir: str):
    """
    打印统计报告
    
    Args:
        stats: 统计结果
        prediction_dir: 目录路径
    """
    print("=" * 70)
    print(f"Thinking格式通过率统计报告")
    print(f"目录: {prediction_dir}")
    print("=" * 70)
    print(f"{'类别':<20} {'总数':<10} {'通过':<10} {'失败':<10} {'通过率':<10}")
    print("-" * 70)
    
    total_all = 0
    passed_all = 0
    failed_all = 0
    
    for category in categories:
        if category in stats:
            data = stats[category]
            total = data["total"]
            passed = data["passed"]
            failed = data["failed"]
            ratio = (passed / total * 100) if total > 0 else 0.0
            
            total_all += total
            passed_all += passed
            failed_all += failed
            
            print(f"{category:<20} {total:<10} {passed:<10} {failed:<10} {ratio:.2f}%")
        else:
            print(f"{category:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("-" * 70)
    overall_ratio = (passed_all / total_all * 100) if total_all > 0 else 0.0
    print(f"{'总计':<20} {total_all:<10} {passed_all:<10} {failed_all:<10} {overall_ratio:.2f}%")
    print("=" * 70)

    # for easy copy
    for category in categories:
        if category in stats:
            data = stats[category]
            total = data["total"]
            passed = data["passed"]
            failed = data["failed"]
            ratio = (passed / total * 100) if total > 0 else 0.0
            print(f"{ratio:.2f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="统计评测文件中prediction字段的thinking格式通过率")
    parser.add_argument(
        "--prediction_dir",
        default="/mnt/shared-storage-user/ailab-hs/zouyicheng/POLAR_Next/opencompass/chat_objective/20260107_210624/predictions/verl_grpo_policy_Qwen3-30B-A3B-MH_reward_RULE_data_chemistry_moi_step_120_retrain",
        type=str,
        help="包含评测文件的目录路径"
    )
    
    args = parser.parse_args()
    
    # 分析并打印报告
    stats = analyze_thinking_ratio(args.prediction_dir)
    if stats:
        print_report(stats, args.prediction_dir)
    else:
        print("未找到有效的评测数据")


if __name__ == "__main__":
    main()
