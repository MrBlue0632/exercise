"""Interactive TinyStories continuation chat for a trained miniLLM checkpoint."""

import argparse
import os
import pathlib
from pathlib import Path

import torch

from model import ModelArgs, Transformer


def load_model(checkpoint_path: Path, device: torch.device) -> Transformer:
    # train.py serializes Path values inside train_args. Make trusted checkpoints
    # portable when opening a Linux-created checkpoint on Windows.
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    if all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    model = Transformer(ModelArgs(**checkpoint["model_args"]))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError("Install sentencepiece before running chat.py") from error
    if not args.checkpoint.is_file() or not args.tokenizer.is_file():
        raise FileNotFoundError("checkpoint or tokenizer model was not found")

    device = torch.device(args.device)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model = load_model(args.checkpoint, device)
    context: list[int] = []

    print("TinyStories interactive continuation. Commands: /clear, /exit")
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/clear":
            context.clear()
            print("context cleared")
            continue

        user_tokens = tokenizer.encode(user_text, out_type=int)
        if not context:
            user_tokens = [tokenizer.bos_id(), *user_tokens]
        context.extend(user_tokens)
        context = context[-model.params.max_seq_len :]
        prompt = torch.tensor([context], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        new_tokens = output[0, prompt.size(1) :].tolist()
        response = tokenizer.decode(new_tokens).strip()
        print(f"model> {response}")
        context = output[0].tolist()[-model.params.max_seq_len :]


if __name__ == "__main__":
    main()
