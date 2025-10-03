# Spiking GPT‑OSS: Brain‑Inspired Spike Encoding for GPT‑OSS 20B

## Overview

This repository provides a simple yet functional wrapper around
OpenAI's **GPT‑OSS** models that adds **spike encoding** to the model
outputs.  The implementation draws inspiration from the
**SpikingBrain** project, which demonstrates how to convert dense
Transformer models into sparse, brain‑inspired versions using a
combination of efficient attention, Mixture‑of‑Experts (MoE) modules
and adaptive spiking mechanisms.  While the full
SpikingBrain pipeline involves continual pre‑training, MoE up‑cycling
and hardware‑specific optimisations, this repository focuses on the
final stage: converting a pre‑trained model's activations into sparse
spike trains.

The wrapper in `spiking_wrapper.py` can be used with any
`AutoModelForCausalLM` compatible model.  By default we test with
`gpt2` because it is small enough to load on a CPU.  To use the
official GPT‑OSS weights you should specify the model identifier
`openai/gpt‑oss‑20b` or `openai/gpt‑oss‑120b`.  These weights can be
downloaded from Hugging Face under the permissive Apache‑2.0 licence.

Three spike encoding schemes are supported:

* **Binary encoding** – values above a threshold fire a 1, otherwise 0.
* **Ternary encoding** – positive values fire +1, negative values fire −1
  and values in between are silent.
* **Bitwise encoding** – activations are quantised and decomposed into
  bit planes, yielding a sparse representation across multiple
  timesteps.

These encodings approximate the *pseudo‑spiking* mechanism used in
SpikingBrain‑7B【710166082177713†L121-L132】 and are useful for exploring
brain‑inspired computation on existing models.

## Installation

Install the required Python packages (PyTorch and HuggingFace
Transformers).  We recommend Python 3.10 or later.

```
pip install torch transformers
```

> **Note:** Running the full GPT‑OSS‑20B model requires a GPU with
> around 16 GB of memory or more.  For CPU‑only environments you can
> test the wrapper using `gpt2` or another small model.

## Downloading GPT‑OSS Weights

OpenAI provides the GPT‑OSS weights under a permissive Apache‑2.0
licence.  The 20B variant is designed to fit on a single 16 GB GPU and
features 21 billion parameters with 3.6 billion active parameters【981994616451650†L24-L28】.
You can download the model via Hugging Face:

* Pre‑trained GPT‑OSS 20B: <https://huggingface.co/openai/gpt-oss-20b>
* Pre‑trained GPT‑OSS 120B: <https://huggingface.co/openai/gpt-oss-120b>

Alternatively, you can install the model and its dependencies with
vLLM:

```
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
vllm serve openai/gpt-oss-20b
```

Refer to the [gpt‑oss repository](https://github.com/openai/gpt-oss) for
more information on Harmony formatting and advanced usage.

## Quick Start

Clone this repository (or copy the files into your project) and run
the provided script:

```
python run_spiking_gpt_oss.py \
    --model gpt2 \
    --encoding binary \
    --threshold 0.0 \
    --prompt "What is the capital of Italy?" \
    --max-new-tokens 32
```

This will load the GPT‑2 model, wrap it with a binary spiking layer
and print the generated answer.  To run GPT‑OSS you would change
`--model` to `openai/gpt-oss-20b` and ensure you have enough GPU
memory.

### Options

* `--encoding {none,binary,ternary,bitwise}` – select the spike
  encoding type.
* `--threshold` – threshold for binary/ternary encoding.
* `--num-bits` – number of bits for bitwise encoding.
* `--use-generation` – if set, use the auto‑regressive `generate()`
  API; otherwise perform a single forward pass and report spiked
  logits.
* `--device` – device on which to run the model (e.g. `cuda`,
  `cuda:0`, `cpu`).  If omitted the model uses the default device.

## How It Works

The core logic resides in `spiking_wrapper.py`.  The
`SpikingBrainWrapper` class accepts a pre‑trained causal language
model and a `SpikeEncodingConfig` describing how to convert its
activations to spikes.  During the forward pass, the wrapper calls
the base model's `forward` method to obtain dense logits and then
applies the chosen spike encoding.  For generation the wrapper can
optionally return spike‑encoded scores at each decoding step.

The encoding schemes are simplified approximations of the adaptive
threshold spiking described in the SpikingBrain report.
By experimenting with different thresholds and bit depths you can
explore the trade‑offs between sparsity, information content and
inference behaviour.

## Limitations and Future Work

* **No continual pre‑training:** This wrapper does not implement the
  multi‑stage conversion pipeline or MoE up‑cycling used by
  SpikingBrain.  It simply applies a spike encoding on top of a
  frozen base model.  Achieving the performance reported by
  SpikingBrain would require re‑training with hybrid attention,
  spike‑aware optimisation and expert replication【338975382826362†L182-L190】.
* **Memory requirements:** Loading GPT‑OSS‑20B or larger models
  requires significant GPU memory.  For local experimentation use
  smaller models or vLLM with streaming quantisation.
* **Bitwise encoding semantics:** The bitwise encoding implemented
  here is one of several possible encodings.  The Int2Spike
  repository describes binary, ternary and bitwise encodings in
  detail; more sophisticated schemes could be explored.

Despite these limitations, this repository provides a functional
starting point for researchers interested in exploring brain‑inspired
representations on modern LLMs.
