import runpod
from llmlingua import PromptCompressor
import torch

llm_lingua = PromptCompressor(
    model_name="models/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)
torch.set_default_device("cuda")


def handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]

    context = job_input.get("context", [""])
    instruction = job_input.get("instruction", "")
    question = job_input.get("question", "")
    rate = job_input.get("rate", 0.5)
    force_tokens = job_input.get("force_tokens", ["\n", "?", "<SEP>"])

    compressed_prompt = llm_lingua.compress_prompt(
        context=context,
        instruction=instruction,
        question=question,
        rate=rate,
        force_tokens=force_tokens,
    )
    return compressed_prompt


runpod.serverless.start({"handler": handler})
