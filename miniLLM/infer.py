import argparse
from pathlib import Path

import torch

from model import ModelArgs, Transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token IDs with a trained miniLLM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=int, nargs="+", required=True, help="Prompt token IDs")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli_args = parser.parse_args()

    # Checkpoints created by train.py also contain optimizer/configuration data,
    # so load trusted local checkpoints normally rather than weights_only mode.
    checkpoint = torch.load(cli_args.checkpoint, map_location=cli_args.device, weights_only=False)
    model = Transformer(ModelArgs(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    # torch.compile wraps modules and prefixes state-dict keys with _orig_mod.
    if all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.to(cli_args.device).eval()

    prompt = torch.tensor([cli_args.prompt], dtype=torch.long, device=cli_args.device)
    generated = model.generate(
        prompt,
        max_new_tokens=cli_args.max_new_tokens,
        temperature=cli_args.temperature,
        top_k=cli_args.top_k,
    )

    print(generated[0].tolist())


if __name__ == "__main__":
    main()
