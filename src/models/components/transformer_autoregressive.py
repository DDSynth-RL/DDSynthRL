"""Autoregressive Transformer over a frozen synth token space."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.synth_backends.dexed.dexed_metadata import FIRST_OPERATOR_INDEX, OPERATOR_STRIDE
from src.models.components.position_encoding import PositionalEncoding1D, PositionEmbeddingSine
from src.models.components.transformer_layers import Transformer
from src.models.components.transformer_non_autoregressive import CNNBackbone
from src.models.synth_token_space import SynthTokenSpace
from src.utils.probability import GaussianKernelConv


class AutoregressiveParamTransformer(nn.Module):
    """Decoder Transformer that predicts synth tokens sequentially with a causal mask."""

    def __init__(
        self,
        *,
        token_space: SynthTokenSpace,
        synth: str,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = True,
        gaussian_sigma: float = 0.02,
    ) -> None:
        super().__init__()
        self.token_space = token_space
        self.synth = str(synth).lower()
        self.order = list(token_space.order)
        self.fields = list(token_space.fields)
        self.cardinals = list(token_space.cardinalities)
        self.seq_len = len(self.cardinals)
        self.max_card = max(self.cardinals)

        self.full_indices: List[int | None] = [field.full_index for field in self.fields]
        self.is_midi: List[bool] = [field.is_midi for field in self.fields]
        self.numeric_positions = {
            idx for idx, field in enumerate(self.fields) if field.is_param and field.mode == "num"
        }

        self.operator_for_token: List[int | None] = [None] * self.seq_len
        self.op_output_token_index: List[int | None] = [None] * 6
        self._maskable_rel_indices = {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        }
        if self.synth == "dexed":
            for token_idx, field in enumerate(self.fields):
                full_idx = field.full_index
                if full_idx is None:
                    continue
                if not (FIRST_OPERATOR_INDEX <= full_idx < FIRST_OPERATOR_INDEX + 6 * OPERATOR_STRIDE):
                    continue
                rel = (full_idx - FIRST_OPERATOR_INDEX) % OPERATOR_STRIDE
                op = (full_idx - FIRST_OPERATOR_INDEX) // OPERATOR_STRIDE
                if rel == 8:
                    self.op_output_token_index[op] = token_idx
                if rel in self._maskable_rel_indices:
                    self.operator_for_token[token_idx] = op

        self.backbone = CNNBackbone(in_channels=1, d_model=d_model)
        self.enc_pos_embed = PositionEmbeddingSine(d_model // 2)
        self.query_pos_embed = PositionalEncoding1D(d_model, self.seq_len)
        self.transformer = Transformer(
            num_queries=self.seq_len,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )

        self.token_embed = nn.Embedding(self.max_card, d_model)
        self.proj = nn.Linear(d_model, self.max_card)
        self.dropout = nn.Dropout(float(dropout))
        self.gaussian_conv = GaussianKernelConv(sigma=gaussian_sigma)
        self.loss_log_vars = nn.Parameter(torch.zeros(self.seq_len))

        cardinals_t = torch.tensor(self.cardinals, dtype=torch.long)
        class_ids = torch.arange(self.max_card, dtype=torch.long)[None, :].expand(self.seq_len, -1)
        invalid = class_ids >= cardinals_t[:, None]
        self.register_buffer("invalid_class_mask", invalid, persistent=False)

    def forward(self, spectrogram: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.seq_len:
            raise ValueError(f"Expected token_ids shape (B,{self.seq_len}), got {tuple(token_ids.shape)}")

        batch = token_ids.size(0)
        feats = self.backbone(spectrogram)
        pos_enc = self.enc_pos_embed(feats)

        bos = torch.zeros((batch, 1), dtype=token_ids.dtype, device=token_ids.device)
        dec_tokens = torch.cat([bos, token_ids[:, :-1]], dim=1)
        tgt = self.token_embed(dec_tokens).transpose(0, 1)

        causal = torch.triu(torch.ones((self.seq_len, self.seq_len), device=tgt.device), diagonal=1)
        causal = causal.masked_fill(causal == 1, float("-inf"))

        src = feats.flatten(2).permute(2, 0, 1)
        src_pos = pos_enc.flatten(2).permute(2, 0, 1)
        memory = self.transformer.encoder(src, pos=src_pos)

        query_pos = (
            self.query_pos_embed.encoding[0, : self.seq_len, :]
            .unsqueeze(1)
            .repeat(1, batch, 1)
            .to(tgt.device)
        )
        dec_out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            pos=src_pos,
            query_pos=query_pos,
        )
        if dec_out.dim() == 4:
            dec_out = dec_out[-1]
        dec_out = dec_out.transpose(0, 1)
        logits = self.proj(self.dropout(dec_out))
        invalid = self.invalid_class_mask.to(device=logits.device)
        return logits.masked_fill(invalid.unsqueeze(0), float("-inf"))

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        midi_label_smoothing: float = 0.0,
        midi_label_smoothing_apply: Sequence[bool] | None = None,
        token_weights: torch.Tensor | None = None,
        uncertainty_weighting: bool = False,
        return_components: bool = False,
        return_midi_per_head: bool = False,
        return_per_param_loss: bool = False,
    ):
        device = logits.device
        if targets.shape[:2] != logits.shape[:2]:
            raise ValueError(f"targets shape {tuple(targets.shape)} must match logits {tuple(logits.shape[:2])}")

        if token_weights is not None:
            if not isinstance(token_weights, torch.Tensor):
                token_weights = torch.as_tensor(token_weights, dtype=logits.dtype, device=device)
            else:
                token_weights = token_weights.to(device=device, dtype=logits.dtype)
            if token_weights.ndim != 1 or int(token_weights.numel()) != int(self.seq_len):
                raise ValueError(
                    "token_weights must be a 1D tensor with length equal to the token sequence length "
                    f"({self.seq_len}), got shape {tuple(token_weights.shape)}"
                )
            if (token_weights < 0).any():
                raise ValueError("token_weights must be non-negative.")

        param_losses: List[tuple[int, torch.Tensor]] = []
        midi_losses: List[tuple[int, torch.Tensor]] = []
        midi_per_head_losses: List[torch.Tensor] = []
        per_token_loss_stats: List[tuple[str, torch.Tensor, int]] = []
        token_losses: List[tuple[int, torch.Tensor]] = []

        midi_apply = list(midi_label_smoothing_apply) if midi_label_smoothing_apply is not None else []
        op_enabled: List[torch.Tensor | None] = [None] * 6
        if self.synth == "dexed":
            for op in range(6):
                token_idx = self.op_output_token_index[op]
                if token_idx is None:
                    continue
                vol_card = self.cardinals[token_idx]
                vol_targets = targets[:, token_idx]
                if vol_card <= 1:
                    continue
                vol_values = vol_targets.float() / float(max(vol_card - 1, 1))
                op_enabled[op] = vol_values >= 1e-3

        midi_head = 0
        for i, cardinal in enumerate(self.cardinals):
            if cardinal <= 1:
                continue

            valid_mask = None
            operator_id = self.operator_for_token[i]
            if operator_id is not None and op_enabled[operator_id] is not None:
                valid_mask = op_enabled[operator_id]
                if not valid_mask.any():
                    continue

            step_logits = logits[:, i, :cardinal]
            step_targets = targets[:, i]
            if step_targets.numel() > 0:
                min_idx = int(step_targets.min().item())
                max_idx = int(step_targets.max().item())
                if min_idx < 0 or max_idx >= cardinal:
                    raise RuntimeError(
                        f"Class index out of range for token '{self.order[i]}' "
                        f"(index {i}, cardinal {cardinal}): min={min_idx}, max={max_idx}"
                    )

            target_probs = F.one_hot(step_targets, num_classes=cardinal).float()
            if self.is_midi[i]:
                if midi_label_smoothing > 0:
                    apply_gaussian = midi_head < len(midi_apply) and bool(midi_apply[midi_head])
                    if apply_gaussian:
                        target_probs = torch.zeros_like(target_probs)
                        target_probs.scatter_(1, step_targets.unsqueeze(1), 1.0)
                        target_probs = self.gaussian_conv(target_probs)
                        target_probs = F.normalize(target_probs, p=1, dim=-1)
                    else:
                        smooth = midi_label_smoothing / float(max(cardinal - 1, 1))
                        target_probs = torch.full_like(target_probs, smooth)
                        target_probs.scatter_(1, step_targets.unsqueeze(1), 1.0 - midi_label_smoothing)
            elif i in self.numeric_positions and label_smoothing > 0:
                target_probs = self.gaussian_conv(target_probs)

            log_probs = F.log_softmax(step_logits / float(temperature), dim=-1)
            if valid_mask is not None:
                log_probs = log_probs[valid_mask]
                target_probs = target_probs[valid_mask]

            if target_probs.shape != log_probs.shape:
                raise RuntimeError(
                    f"Shape mismatch for token '{self.order[i]}' (index {i}, cardinal {cardinal}): "
                    f"target_probs {target_probs.shape}, log_probs {log_probs.shape}"
                )
            if target_probs.numel() == 0:
                continue

            loss_val = -(target_probs * log_probs).sum(dim=-1).mean()
            token_losses.append((i, loss_val))

            if return_per_param_loss:
                token_count = int(valid_mask.sum().item()) if valid_mask is not None else int(step_targets.shape[0])
                per_token_loss_stats.append((self.order[i], loss_val, token_count))

            if self.is_midi[i]:
                midi_losses.append((i, loss_val))
                midi_per_head_losses.append(loss_val)
                midi_head += 1
            else:
                param_losses.append((i, loss_val))

        if not token_losses:
            total_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
        elif token_weights is None:
            if uncertainty_weighting:
                weighted_terms = []
                for token_idx, loss_val in token_losses:
                    s = self.loss_log_vars[token_idx]
                    weighted_terms.append(torch.exp(-s) * loss_val + s)
                total_loss = torch.stack(weighted_terms).mean()
            else:
                total_loss = torch.stack([loss for _, loss in token_losses]).mean()
        elif uncertainty_weighting:
            terms = []
            weights = []
            for token_idx, loss_val in token_losses:
                s = self.loss_log_vars[token_idx]
                terms.append(torch.exp(-s) * loss_val + s)
                weights.append(token_weights[token_idx])
            terms_t = torch.stack(terms)
            weights_t = torch.stack(weights)
            total_loss = (terms_t * weights_t).sum() / (weights_t.sum() + 1e-12)
        else:
            losses_t = torch.stack([loss for _, loss in token_losses])
            weights_t = torch.stack([token_weights[token_idx] for token_idx, _ in token_losses])
            total_loss = (losses_t * weights_t).sum() / (weights_t.sum() + 1e-12)

        if not param_losses:
            param_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
        elif token_weights is None:
            param_loss = torch.stack([loss for _, loss in param_losses]).mean()
        else:
            losses_t = torch.stack([loss for _, loss in param_losses])
            weights_t = torch.stack([token_weights[token_idx] for token_idx, _ in param_losses])
            param_loss = (losses_t * weights_t).sum() / (weights_t.sum() + 1e-12)

        if not midi_losses:
            midi_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
        elif token_weights is None:
            midi_loss = torch.stack([loss for _, loss in midi_losses]).mean()
        else:
            losses_t = torch.stack([loss for _, loss in midi_losses])
            weights_t = torch.stack([token_weights[token_idx] for token_idx, _ in midi_losses])
            midi_loss = (losses_t * weights_t).sum() / (weights_t.sum() + 1e-12)

        if return_components and return_midi_per_head and return_per_param_loss:
            return (
                total_loss,
                param_loss,
                midi_loss,
                torch.stack(midi_per_head_losses) if midi_per_head_losses else torch.tensor([], device=device),
                per_token_loss_stats,
            )
        if return_components and return_midi_per_head:
            return (
                total_loss,
                param_loss,
                midi_loss,
                torch.stack(midi_per_head_losses) if midi_per_head_losses else torch.tensor([], device=device),
            )
        if return_components and return_per_param_loss:
            return total_loss, param_loss, midi_loss, per_token_loss_stats
        if return_components:
            return total_loss, param_loss, midi_loss
        return total_loss
