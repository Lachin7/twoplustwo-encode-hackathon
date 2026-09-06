"""Same baseline through Tinker: a base model, or your fine-tuned sampler checkpoint.

    uv sync --extra tinker
    uv run baseline/tinker_predict.py --out-dir submissions/qwen38-27b --base-model Qwen/Qwen3.8-27B --ids 13-1,51-12
    uv run baseline/tinker_predict.py --out-dir submissions/mine --base-model Qwen/Qwen3.8-27B \
        --model-path tinker://<run-id>/sampler_weights/final

Needs TINKER_API_KEY and TINKER_PROJECT_ID in .env. Hackathon model is Qwen/Qwen3.8-27B.
Writes the same files as llm_predict.py.
"""

import argparse
import asyncio
from pathlib import Path

import tinker
from common import FORMAT_HINT, SYSTEM_PROMPT, load_env, parse_ids, run, selected_tasks
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

from sb import DEFAULT_DATASET


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    p.add_argument("--ids", help="comma-separated task ids (default: all)")
    p.add_argument("--base-model", required=True, help="e.g. Qwen/Qwen3.8-27B")
    p.add_argument("--model-path", help="tinker://... sampler checkpoint. Omit to sample the base model.")
    p.add_argument(
        "--renderer",
        help="override chat renderer. Default: cookbook recommended (qwen3_8_xhigh_reasoning for Qwen3.8).",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=16384, help="xhigh reasoning + sheet JSON need headroom")
    p.add_argument("--resume", action="store_true", help="skip ids already in predictions.jsonl")
    p.add_argument(
        "--agent",
        choices=("off", "auto", "always"),
        default="auto",
        help="off=one-shot only. auto=Python loop when >20 graded cells (default). always=loop on every task.",
    )
    return p.parse_args()


async def main():
    load_env()
    args = parse_args()
    sampler = tinker.ServiceClient().create_sampling_client(base_model=args.base_model, model_path=args.model_path)
    renderer_name = args.renderer or get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, get_tokenizer(args.base_model))
    params = types.SamplingParams(max_tokens=args.max_tokens, temperature=0, stop=renderer.get_stop_sequences())

    def _content(response) -> str:
        content = renderer.parse_response(response.sequences[0].tokens)[0]["content"]
        if not isinstance(content, str):
            content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
        return content

    async def complete_chat(messages):
        model_input = renderer.build_generation_prompt(messages)
        response = await sampler.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
        return _content(response), model_input.length, len(response.sequences[0].tokens)

    async def complete(prompt: str):
        return await complete_chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + FORMAT_HINT}]
        )

    tasks = selected_tasks(Path(args.dataset_dir), parse_ids(args.ids))
    label = args.model_path or args.base_model
    await run(
        complete,
        f"{label} renderer={renderer_name}",
        tasks,
        Path(args.out_dir),
        args.concurrency,
        resume=args.resume,
        agent=args.agent,
        complete_chat=complete_chat,
    )


if __name__ == "__main__":
    asyncio.run(main())
