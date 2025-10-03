"""A simple spiking wrapper around HuggingFace causal language models.

This module defines a wrapper class that can be used to wrap any
`AutoModelForCausalLM` compatible model and convert its floating‑point
activations into sparse spike trains.  The idea is inspired by the
conversion pipeline described in the SpikingBrain technical report【338975382826362†L196-L204】, but
implemented here in a minimal form for demonstration purposes.

At a high level the wrapper performs the following steps:

1.  Loads a base model from HuggingFace.  For our use case the base
    model should be `openai/gpt‑oss‑20b`, but any causal language model
    can be used (e.g. GPT‑2 for quick testing on machines with limited
    memory).
2.  Exposes the standard `forward` and `generate` APIs, delegating to
    the underlying model for dense computation.
3.  After the dense output is computed, applies a spike encoding
    function to convert activations to sparse binary/ternary/bitwise
    representations.  This optional step can be disabled if one wants
    to use the original floating‑point logits.

The spike encoding implemented here is intentionally simple.  The
SpikingBrain paper proposes an adaptive threshold spiking mechanism
where the membrane potential is quantized into integer spike counts and
expanded into spike trains【338975382826362†L196-L204】.  In this wrapper we support
three basic encodings:

* **binary:** encodes each activation as 1 if it is above a threshold
  and 0 otherwise.
* **ternary:** encodes activations above a positive threshold as +1,
  activations below the negative threshold as −1 and everything
  between as 0.
* **bitwise:** quantizes activations to an integer within a limited
  range and then decomposes the integer into a list of bits.

These encodings are meant to approximate the pseudo‑spiking behaviour
used in the SpikingBrain code base【710166082177713†L121-L132】.  They provide
sparse, event‑driven representations which can be further processed by
specialised hardware or algorithms.  For this demonstration they simply
serve to illustrate how one might hook into a HuggingFace model.

The wrapper does not alter the internal layers of the base model nor
perform continual pre‑training.  The heavy lifting of converting a
dense model into a spiking model would normally require re‑training
and architecture modifications as described in the SpikingBrain report.
Nevertheless, this code can be used as a starting point for
experimenting with spiking encodings on top of existing models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class SpikeEncodingConfig:
    """Configuration for spike encoding.

    Attributes
    ----------
    type : str
        The type of spike encoding to use.  Supported values are
        ``"none"``, ``"binary"``, ``"ternary"`` and ``"bitwise"``.
    threshold : float
        Threshold value used by binary and ternary encodings.  Values
        greater than this threshold fire a spike (and values less than
        the negative threshold fire a negative spike for ternary).
    num_bits : int
        Number of bits to use for bitwise encoding.  Ignored for
        other encoding types.
    clamp_min : float
        Minimum value to clamp activations to before quantisation in
        bitwise encoding.
    clamp_max : float
        Maximum value to clamp activations to before quantisation in
        bitwise encoding.
    """

    type: str = "none"
    threshold: float = 0.0
    num_bits: int = 8
    clamp_min: float = -3.0
    clamp_max: float = 3.0

    def __post_init__(self) -> None:
        valid_types = {"none", "binary", "ternary", "bitwise"}
        if self.type not in valid_types:
            raise ValueError(f"Invalid spike encoding type {self.type}. "
                             f"Supported types: {valid_types}")


class SpikingBrainWrapper(PreTrainedModel):
    """A wrapper that adds spike encoding to the outputs of a causal model.

    Parameters
    ----------
    base_model : PreTrainedModel
        A causal language model compatible with the HuggingFace API (e.g.
        GPT‑2, Qwen, GPT‑OSS).  This model performs all dense
        computation.
    encoding_config : SpikeEncodingConfig
        Configuration describing the spike encoding to apply to the
        activations/logits.  If ``type`` is ``"none"`` no spiking is
        applied and the outputs of the base model are returned as is.
    device : Union[str, torch.device], optional
        Device on which to place the model and perform inference.  If
        ``None`` the model will remain on its current device.

    Notes
    -----
    The wrapper delegates all attribute accesses not explicitly
    implemented here to the underlying model.  This allows it to be
    used seamlessly wherever the base model would be used.  Only the
    forward method is intercepted to apply spike encoding.
    """

    config: SpikeEncodingConfig  # type: ignore[assignment]

    def __init__(self,
                 base_model: PreTrainedModel,
                 encoding_config: Optional[SpikeEncodingConfig] = None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        # We do not call PreTrainedModel.__init__ because we do not have a
        # config object for this wrapper.  Instead we store the underlying
        # model and encoding configuration.
        object.__setattr__(self, "base_model", base_model)
        if encoding_config is None:
            encoding_config = SpikeEncodingConfig(type="none")
        object.__setattr__(self, "encoding_config", encoding_config)
        # Move model to device if specified
        if device is not None:
            self.base_model.to(device)

    # ------------------------------------------------------------------
    # Spike encoding implementations
    # ------------------------------------------------------------------
    def _binary_encode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Binary spike encoding: convert activations to {0, 1} spikes.

        Any value greater than the configured threshold is mapped to 1
        and all other values are mapped to 0.
        """
        return (tensor > self.encoding_config.threshold).float()

    def _ternary_encode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ternary spike encoding: convert activations to {-1, 0, 1} spikes.

        Values greater than the positive threshold map to +1, values less
        than the negative threshold map to −1 and everything in between
        maps to 0.  The same threshold is used for both directions.
        """
        pos = (tensor > self.encoding_config.threshold).float()
        neg = (tensor < -self.encoding_config.threshold).float()
        return pos - neg

    def _bitwise_encode(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Bitwise spike encoding: quantise and decompose into bits.

        The input tensor is first clamped to the range
        [clamp_min, clamp_max] and normalised to the interval [0, 1].  It
        is then scaled to an integer range [0, 2^num_bits - 1] and
        decomposed into its binary representation.  The result is a
        list of bit planes (each a tensor of 0/1 values) from the
        most significant bit to the least significant bit.  This
        representation can be interpreted as firing a spike on each
        timestep if the corresponding bit is set.
        """
        cfg = self.encoding_config
        # Clamp and normalise
        x = torch.clamp(tensor, cfg.clamp_min, cfg.clamp_max)
        x = (x - cfg.clamp_min) / (cfg.clamp_max - cfg.clamp_min)
        x_int = (x * (2 ** cfg.num_bits - 1)).round().int()
        bits = []
        for i in reversed(range(cfg.num_bits)):
            bits.append(((x_int >> i) & 1).float())
        return bits

    def _apply_spiking(self, tensor: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply the configured spike encoding to a tensor.

        If the encoding type is ``"none"`` the original tensor is returned.
        Otherwise a new tensor or list of tensors representing the spike
        encoded activations is returned.
        """
        enc_type = self.encoding_config.type
        if enc_type == "none":
            return tensor
        if enc_type == "binary":
            return self._binary_encode(tensor)
        if enc_type == "ternary":
            return self._ternary_encode(tensor)
        if enc_type == "bitwise":
            return self._bitwise_encode(tensor)
        raise NotImplementedError(f"Unknown encoding type: {enc_type}")

    # ------------------------------------------------------------------
    # HuggingFace API
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls,
                        model_name_or_path: str,
                        encoding_config: Optional[SpikeEncodingConfig] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        **kwargs) -> "SpikingBrainWrapper":
        """Load a HuggingFace model and wrap it with spike encoding.

        Parameters
        ----------
        model_name_or_path : str
            Identifier of the model to load (e.g. ``"openai/gpt-oss-20b"``).
        encoding_config : SpikeEncodingConfig, optional
            The spike encoding configuration.  Defaults to no encoding.
        device : Union[str, torch.device], optional
            Device on which to load the model.
        **kwargs
            Additional keyword arguments are forwarded to
            `AutoModelForCausalLM.from_pretrained`.

        Returns
        -------
        SpikingBrainWrapper
            A wrapper instance containing the loaded model.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs
        )
        return cls(base_model=base_model, encoding_config=encoding_config, device=device)

    # Delegate attribute lookups to the base model unless overridden
    def __getattr__(self, name: str):  # type: ignore[override]
        return getattr(self.base_model, name)

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Move the base model to a different device and return self."""
        self.base_model = self.base_model.to(*args, **kwargs)
        return self

    def forward(self, *args, **kwargs):  # type: ignore[override]
        """Forward pass through the model with optional spike encoding.

        The method executes the base model's forward pass and applies
        spike encoding to the `logits` in the returned output (if
        present).  The returned dictionary mirrors the structure of the
        base model's output but with spike encoded logits if applicable.
        """
        # Run the base model
        outputs = self.base_model(*args, **kwargs)
        # Apply spike encoding to logits if they exist
        if hasattr(outputs, "logits"):
            spiked = self._apply_spiking(outputs.logits)
            # Using a simple object copy to avoid mutating original outputs
            # Note: for bitwise encoding the spiked result is a list of
            # tensors, so we return it under a new attribute name
            if isinstance(spiked, list):
                # Add a new attribute containing bit planes
                outputs.spike_bits = spiked  # type: ignore[attr-defined]
            else:
                outputs.spiked_logits = spiked  # type: ignore[attr-defined]
        return outputs

    def generate(self, *args, **kwargs):  # type: ignore[override]
        """Generate text with optional spike encoding.

        This method delegates to the base model's ``generate`` method
        and applies spike encoding to the generated logits before
        sampling (if the `return_dict_in_generate` flag is used).  For
        simplicity we currently only support spike encoding on the
        batched logits returned when ``output_scores=True``.  If you
        enable spike encoding but do not request scores the generation
        will proceed normally.
        """
        # If users want to inspect scores, make sure they are returned
        if self.encoding_config.type != "none":
            kwargs.setdefault("output_scores", True)
            kwargs.setdefault("return_dict_in_generate", True)
        out = self.base_model.generate(*args, **kwargs)
        # If we returned a dict and scores are available, apply spiking
        if self.encoding_config.type != "none" and isinstance(out, dict) and "scores" in out:
            spiked_scores: List[Union[torch.Tensor, List[torch.Tensor]]] = []
            for score in out["scores"]:
                spiked_scores.append(self._apply_spiking(score))
            out["spiked_scores"] = spiked_scores
        return out


def load_tokenizer(model_name_or_path: str, **kwargs) -> PreTrainedTokenizerBase:
    """Utility function to load a tokenizer with sensible defaults.

    For causal language models we set ``padding_side`` and
    ``truncation_side`` to ``"left"`` as recommended by the GPT‑OSS
    authors【981994616451650†L71-L76】.  We also ensure that a ``pad_token`` is
    defined by falling back to the ``eos_token`` if necessary.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    # Ensure correct padding and truncation behaviour
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer