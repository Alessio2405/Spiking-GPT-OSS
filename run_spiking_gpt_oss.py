#!/usr/bin/env python3
"""Command line interface for running GPT‑OSS with a spiking wrapper.

This script demonstrates how to load a large language model (LLM)
compatible with HuggingFace's ``AutoModelForCausalLM`` API, wrap it
with the ``SpikingBrainWrapper`` defined in ``spiking_wrapper.py`` and
perform inference.  The default model used for quick testing is
``gpt2`` because it is small enough to fit into limited memory.  To
run the full ``gpt‑oss‑20b`` model you must have access to a GPU with
sufficient memory and specify ``--model openai/gpt-oss-20b``.

Example usage::

    python run_spiking_gpt_oss.py --model openai/gpt-oss-20b \
        --encoding binary --threshold 0.0 \
        --prompt "Hello, how are you?" --max-new-tokens 64

The script supports binary, ternary and bitwise spike encodings as
described in the SpikingBrain technical report【338975382826362†L196-L204】.  The spiking
configuration is controlled via command line flags.

Note: When running very large models such as GPT‑OSS you may wish to
use the ``device_map="auto"`` or ``device_map="balanced"`` options to
shard the model across multiple GPUs.  Refer to the HuggingFace
documentation for details.
"""

import argparse
import logging
from typing import List, Optional

import torch

from spiking_wrapper import (
    SpikeEncodingConfig,
    SpikingBrainWrapper,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a spiking‑wrapped LLM")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help=(
            "HuggingFace model identifier.  Use 'openai/gpt-oss-20b' to run the "
            "full GPT‑OSS model (requires a GPU).  Default: gpt2 for testing."
        ),
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="none",
        choices=["none", "binary", "ternary", "bitwise"],
        help="Spike encoding type to apply."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for binary and ternary encodings."
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=8,
        help="Number of bits for bitwise encoding.  Ignored otherwise."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world!",
        help="Input prompt to generate from."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top‑p (nucleus) sampling parameter."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device on which to run the model (e.g. 'cuda', 'cuda:0', 'cpu'). "
            "If not specified the default device will be used."
        ),
    )
    parser.add_argument(
        "--use-generation",
        action="store_true",
        help="If set, use the generate() API instead of a single forward pass."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    # Load tokenizer
    logger.info(f"Loading tokenizer for model {args.model}")
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)

    # Build spike encoding config
    enc_cfg = SpikeEncodingConfig(
        type=args.encoding,
        threshold=args.threshold,
        num_bits=args.num_bits,
    )

    # Load model and wrap with spiking
    logger.info(f"Loading base model {args.model}")
    # Use device_map=auto to avoid OOM on small machines for gpt2; for gpt‑oss
    # one may need to shard across multiple GPUs.
    base_model_kwargs = {
        "trust_remote_code": True,
    }
    # When testing with gpt2 it's safe to load on CPU.  For large models
    # this may require device_map; we respect the --device argument.
    if args.device is not None:
        base_model_kwargs["device_map"] = None  # disable HF sharding
    # Load the underlying model
    wrapper = SpikingBrainWrapper.from_pretrained(
        args.model,
        encoding_config=enc_cfg,
        device=args.device,
        **base_model_kwargs,
    )

    # Tokenise the input
    logger.info("Preparing input tokens")
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    )
    if args.device is not None:
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

    if args.use_generation:
        # Use generate() API with spiking scores if requested
        logger.info("Running autoregressive generation")
        outputs = wrapper.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # Decode generated text
        generated = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)[0]
        logger.info("Generated text:\n%s", generated)
        # If spiking was enabled, inspect spiked scores
        if args.encoding != "none" and "spiked_scores" in outputs:
            logger.info("Number of decoding steps: %d", len(outputs["spiked_scores"]))
            logger.info("Shape of first spiked score tensor or bit planes: %s",
                        str(outputs["spiked_scores"][0][0].shape) if isinstance(outputs["spiked_scores"][0], list) else outputs["spiked_scores"][0].shape)
    else:
        # Run a single forward pass through the model
        logger.info("Running single forward pass")
        with torch.no_grad():
            out = wrapper(**inputs)
        # The base model returns logits; after spiking there may be an
        # attribute ``spiked_logits`` or ``spike_bits`` on the output
        if args.encoding == "none":
            logger.info("Raw logits shape: %s", out.logits.shape)
        else:
            if hasattr(out, "spiked_logits"):
                logger.info("Spiked logits shape: %s", out.spiked_logits.shape)
            elif hasattr(out, "spike_bits"):
                logger.info("Number of bit planes: %d", len(out.spike_bits))
                logger.info("Shape of first bit plane: %s", out.spike_bits[0].shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)