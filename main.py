"""Prompt-only 与 Prompt+Context 的 NLP 摘要任务对比。

脚本调用指定的 Qwen 模型获取真实回答，并保存提示词、原 始输 出和基础评测指标。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI


MODEL_NAME = "qwen3.7-plus"
BASE_URL = (
    "https://ws-v1b9u69oqlva8qxw.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1"
)
TEMPERATURE = 0.2
MAX_TOKENS = 600
DEFAULT_OUTPUT_PATH = Path("summary_comparison_results.json")


@dataclass(frozen=True)
class SummaryTask:
    """两组提示词共享的任务定义。"""

    topic: str
    audience: str
    target_length: str
    output_schema: str
    source_text: str
    expected_points: tuple[str, ...]


TASK = SummaryTask(
    topic="生成式AI在医疗场景中的实际效果与主要风险",
    audience="医院管理者",
    target_length="120-180个汉字",
    output_schema="结论、证据、风险三个字段",
    source_text=dedent(
        """
        2024年3月至12月，华东某医疗集团在12家医院的6个科室开展生成式AI试点，
        共服务8000名患者。试点系统用于门诊分诊建议、影像报告初稿和出院小结草拟。
        结果显示，分诊平均等待时间由18分钟降至13分钟，下降约27%；影像报告初稿
        完成时间缩短35%。医生对AI建议的采纳率为59%，其余41%经过修改或被拒绝。
        系统上线后记录到每1万次输出中约7次事实错误，其中2次可能影响临床判断，
        均被医生复核拦截。系统采用院内私有化部署，审计日志保留180天；92%的患者
        同意数据用于试点，8%选择退出。研究者指出，试点仅来自单一医疗集团，样本量
        有限，尚未评估再入院率、误诊率等临床终点，因此不能据此直接推断大规模应用效果。
        """
    ).strip(),
    expected_points=(
        "试点覆盖12家医院、6个科室和8000名患者",
        "分诊等待时间下降约27%，影像报告初稿时间缩短35%",
        "医生采纳率为59%，41%的建议被修改或拒绝",
        "每1万次输出约7次事实错误，其中2次可能影响临床判断",
        "采用私有化部署，审计日志保留180天",
        "样本局限于单一医疗集团，尚未评估临床终点",
    ),
)


def build_prompt_only(task: SummaryTask) -> str:
    """A组：只提供任务指令，不提供外部材料。"""

    return dedent(
        f"""
        你是一名医疗行业研究助理。请基于你已有的知识，总结：
        “{task.topic}”

        目标读者：{task.audience}
        输出长度：{task.target_length}
        输出格式：严格分为“结论”“证据”“风险”三个字段，每个字段各写一段。
        内容要求：突出应用价值、可验证依据和主要限制。
        """
    ).strip()


def build_prompt_with_context(task: SummaryTask) -> str:
    """B组：保留相同任务指令，并加入待总结的材料。"""

    return "\n".join(
        (
            "你是一名医疗行业研究助理。请总结：",
            f"“{task.topic}”",
            "",
            "[参考材料开始]",
            task.source_text,
            "[参考材料结束]",
            "",
            f"目标读者：{task.audience}",
            f"输出长度：{task.target_length}",
            "输出格式：严格分为“结论”“证据”“风险”三个字段，每个字段各写一段。",
            "内容要求：严格依据参考材料作答，不得补充材料中没有的数据或事实；材料未说明",
            "的内容标注“材料未说明”。优先保留样本量、时间变化、比例、限制条件等关键信息。",
        )
    )


NUMBER_EQUIVALENTS = {
    "十二家": "12家",
    "六个科室": "6个科室",
    "八千名": "8000名",
    "百分之二十七": "27%",
    "百分之三十五": "35%",
    "百分之五十九": "59%",
    "百分之四十一": "41%",
    "一万次": "1万次",
    "万次": "1万次",
    "七次": "7次",
    "两次": "2次",
    "一百八十天": "180天",
}

FACT_CHECKS = {
    "试点规模": ("12家", "6个科室", "8000名"),
    "流程效率": ("27%", "35%"),
    "医生采纳情况": ("59%", "41%"),
    "事实错误情况": ("1万次", "7次", "2次"),
    "部署与审计": ("私有化", "180天"),
    "效果限制": ("单一", "临床终点"),
}


def call_model(
    client: OpenAI,
    prompt: str,
    *,
    retries: int = 2,
) -> tuple[str, dict[str, Any]]:
    """调用模型并返回正文及用量信息。"""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("模型返回了空内容")

            usage = response.usage
            usage_data = (
                usage.model_dump()
                if usage is not None and hasattr(usage, "model_dump")
                else {}
            )
            return content.strip(), usage_data
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)

    raise RuntimeError(f"模型调用失败：{last_error}") from last_error


def evaluate_answer(answer: str) -> dict[str, Any]:
    """生成可复核的基础指标，不替代人工语义评价。"""

    compact_answer = "".join(answer.split())
    normalized_answer = compact_answer
    for source, target in NUMBER_EQUIVALENTS.items():
        normalized_answer = normalized_answer.replace(source, target)

    hit_facts = {
        name: all(keyword in normalized_answer for keyword in keywords)
        for name, keywords in FACT_CHECKS.items()
    }
    return {
        "character_count": len(compact_answer),
        "normalized_answer": normalized_answer,
        "has_required_fields": all(
            field in normalized_answer for field in ("结论", "证据", "风险")
        ),
        "fact_coverage": hit_facts,
        "covered_fact_count": sum(hit_facts.values()),
        "total_fact_count": len(FACT_CHECKS),
    }


def create_client() -> OpenAI:
    """从环境变量读取密钥，避免把密钥写入源码。"""

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。")
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120.0)


def run_experiment(client: OpenAI, runs: int) -> list[dict[str, Any]]:
    """用相同参数分别执行 Prompt-only 和 Prompt+Context。"""

    variants = (
        ("Prompt-only", build_prompt_only(TASK)),
        ("Prompt+Context", build_prompt_with_context(TASK)),
    )
    results: list[dict[str, Any]] = []

    for variant_name, prompt in variants:
        for run_index in range(1, runs + 1):
            answer, usage = call_model(client, prompt)
            results.append(
                {
                    "variant": variant_name,
                    "run": run_index,
                    "prompt": prompt,
                    "answer": answer,
                    "metrics": evaluate_answer(answer),
                    "usage": usage,
                }
            )

    return results


def print_actual_outputs(results: list[dict[str, Any]]) -> None:
    for result in results:
        print(f"\n[{result['variant']} | 第 {result['run']} 次]")
        print(result["answer"])


def save_results(
    results: list[dict[str, Any]],
    output_path: Path,
    runs: int,
) -> None:
    payload = {
        "metadata": {
            "model": MODEL_NAME,
            "base_url": BASE_URL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "runs_per_group": runs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "task": asdict(TASK),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 摘要提示词对比实验")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每组运行次数，默认 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"结果文件路径，默认 {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必须大于或等于 1")
    return args


def main() -> None:
    args = parse_args()

    client = create_client()
    results = run_experiment(client, args.runs)
    print_actual_outputs(results)
    save_results(results, args.output, args.runs)


if __name__ == "__main__":
    main()


