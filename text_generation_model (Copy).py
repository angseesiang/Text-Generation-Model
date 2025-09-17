#!/usr/bin/env python3
"""
text_generation_model.py

A minimal GPT-2 text generation script with:
- Class API (TextGenerationModel)
- Command-line interface: --prompt "..." [--sample --temperature 0.8 --top_p 0.9 ...]
- Optional save/load of a local model directory (default: ./model)

Usage examples:
  python text_generation_model.py --prompt "Once upon a time"
  python text_generation_model.py --prompt "Once upon a time" --sample --temperature 0.9 --top_p 0.95
  python text_generation_model.py --prompt "Hello" --save            # saves model/tokenizer to ./model
  python text_generation_model.py --prompt "Hello" --load            # loads from ./model if present
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TextGenerationModel:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Initialize tokenizer & model.

        Args:
            model_name: HF model id or local directory.
            device: "cuda", "cpu", "mps", "auto" or None (auto-detect).
            trust_remote_code: pass through to transformers (usually keep False).
        """
        self.model_name = model_name

        if device in (None, "auto"):
            if torch.cuda.is_available():
                device = "cuda"
            # Apple Silicon, optional:
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Load tokenizer & model (from hub or local dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        # ensure pad token for sampling to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        extra_generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Seed text.
            max_new_tokens: Number of new tokens to generate.
            sample: If True, use sampling; otherwise greedy/beam defaults.
            temperature: Sampling temperature (only used if sample=True).
            top_p: Nucleus sampling probability mass (if sample=True).
            top_k: Top-k sampling (if sample=True).
            extra_generate_kwargs: Dict to forward to `model.generate`.

        Returns:
            Decoded text string (prompt + completion).
        """
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=max(1e-6, float(temperature)),
                    top_p=float(top_p),
                    top_k=int(top_k),
                )
            )
        else:
            gen_kwargs.update(dict(do_sample=False))

        if extra_generate_kwargs:
            gen_kwargs.update(extra_generate_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, save_dir: str = "model") -> None:
        """
        Save model & tokenizer to a local directory.
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Keep any .gitkeep file intact.
        self.tokenizer.save_pretrained(str(path))
        self.model.save_pretrained(str(path))

    @classmethod
    def load(cls, load_dir: str = "model", device: Optional[str] = None) -> "TextGenerationModel":
        """
        Load a previously saved model/tokenizer directory.

        Args:
            load_dir: Path created by `save()`.
            device: Device override (same options as __init__).

        Returns:
            TextGenerationModel instance.
        """
        if not Path(load_dir).exists():
            raise FileNotFoundError(
                f"Model directory '{load_dir}' not found. Run with --save first or specify --model_name."
            )
        return cls(model_name=load_dir, device=device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 text generation")
    p.add_argument("--prompt", required=True, help="Seed text to start generation")
    p.add_argument("--max_new_tokens", type=int, default=50, help="Number of new tokens to generate")
    p.add_argument("--sample", action="store_true", help="Use sampling instead of greedy")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p")
    p.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    p.add_argument("--model_name", default="gpt2", help="HF model id or local dir (default: gpt2)")
    p.add_argument("--save", action="store_true", help="After generation, save to --save_dir")
    p.add_argument("--load", action="store_true", help="Load model from --save_dir before generation")
    p.add_argument("--save_dir", default="model", help="Directory to save/load the model")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load from local dir if requested; else use model_name (hub or directory)
    if args.load:
        model = TextGenerationModel.load(load_dir=args.save_dir, device=args.device)
    else:
        model = TextGenerationModel(model_name=args.model_name, device=args.device)

    text = model.generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print(text)

    # Optionally save the model/tokenizer to local directory
    if args.save:
        model.save(args.save_dir)


if __name__ == "__main__":
    # Suppress excessive TF logs if tf-keras is installed via requirements
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

