"""Direct standard Transformer baseline over the shared raw sequence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

TRAINING_ONLY_PREFIXES = (
    "entity_head",
    "query_role_head",
    "relation_anchor_head",
    "pointer_query",
    "source_key",
    "target_key",
    "boundary_value_head",
    "physical_heads",
)


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    max_tokens: int
    query_token_id: int
    candidate_a_token_id: int
    candidate_b_token_id: int
    hidden_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.0
    answer_count: int = 3


def sinusoidal_positions(length: int, width: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(1, width))
    )
    encoding = torch.zeros((length, width), device=device)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
    return encoding


class DirectTransformer(nn.Module):
    """QKV encoder with a direct answer path and training-only label heads.

    The auxiliary heads make the training information symmetric with ERGT.
    They never feed the direct answer readout and are not called by ``forward``
    during evaluation.
    """

    receives_only_raw_tokens = True
    uses_standard_qkv_attention = True
    has_external_program_or_solver = False
    uses_training_only_matched_supervision = True
    auxiliary_heads_feed_answer_path = False

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.readout_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.readout_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        # A shared candidate scorer aligns answer classes with the two explicit
        # raw query markers. It remains a neural readout, not a path executor.
        self.answer_head = nn.Sequential(
            nn.LayerNorm(3 * config.hidden_dim),
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.abstention_head = nn.Sequential(
            nn.LayerNorm(2 * config.hidden_dim),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # These heads consume the same training labels used by ERGT, but their
        # outputs are never routed into ``answer_head`` or any executor.
        self.entity_head = nn.Linear(config.hidden_dim, 1)
        self.query_role_head = nn.Linear(config.hidden_dim, 6)
        self.relation_anchor_head = nn.Linear(config.hidden_dim, 4)
        self.pointer_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.source_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.target_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.boundary_value_head = nn.Linear(config.hidden_dim, 3)
        self.physical_heads = nn.ModuleDict(
            {
                "action": nn.Linear(config.hidden_dim, 3),
                "cone": nn.Linear(config.hidden_dim, 2),
                "transport": nn.Linear(config.hidden_dim, 2),
                "boundary": nn.Linear(config.hidden_dim, 2),
                "terminal": nn.Linear(config.hidden_dim, 2),
            }
        )

    def _encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.shape != attention_mask.shape or token_ids.dim() != 2:
            raise ValueError("token ids and mask must have matching [B, T] shape")
        if token_ids.size(1) > self.config.max_tokens:
            raise ValueError("input exceeds configured Transformer context")
        hidden = self.embedding(token_ids)
        hidden = hidden + sinusoidal_positions(
            token_ids.size(1), self.config.hidden_dim, token_ids.device
        ).unsqueeze(0)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        return self.norm(hidden)

    def _direct_answer(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        def marker_position(token_id: int, name: str) -> torch.Tensor:
            marker = token_ids.eq(int(token_id)) & attention_mask.bool()
            if not bool(marker.sum(dim=1).eq(1).all()):
                raise ValueError(f"{name} token must be unique in every example")
            return marker.to(torch.int64).argmax(dim=1)

        def state_at(position: torch.Tensor) -> torch.Tensor:
            batch = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch, position]

        query_position = marker_position(self.config.query_token_id, "query")
        candidate_a_position = marker_position(self.config.candidate_a_token_id, "candidate_a")
        candidate_b_position = marker_position(self.config.candidate_b_token_id, "candidate_b")
        for position, name in (
            (candidate_a_position, "candidate_a"),
            (candidate_b_position, "candidate_b"),
        ):
            endpoint_position = position + 1
            valid_endpoint = endpoint_position.lt(token_ids.size(1)) & torch.gather(
                attention_mask.bool(), 1, endpoint_position[:, None]
            ).squeeze(1)
            if not bool(valid_endpoint.all()):
                raise ValueError(f"{name} must be followed by a raw endpoint token")

        query = state_at(query_position)
        # The query grammar itself places the endpoint entity immediately after
        # each candidate marker. Reading that adjacent raw token is equivalent
        # to using class/query tokens in a conventional Transformer; it is not
        # a graph pointer or a supplied program.
        candidate_a = state_at(candidate_a_position) + state_at(candidate_a_position + 1)
        candidate_b = state_at(candidate_b_position) + state_at(candidate_b_position + 1)
        scale = math.sqrt(float(self.config.hidden_dim))
        raw_embeddings = self.embedding(token_ids)
        identity_keys = F.normalize(self.readout_key(raw_embeddings), dim=-1)

        def candidate_pool(position: torch.Tensor) -> torch.Tensor:
            # Tied content attention preserves equality between the endpoint
            # symbol in the raw query and its occurrences in the fact stream.
            # It supplies no edge, path, physical value, or answer label.
            endpoint_identity = state_at(position + 1)
            raw_endpoint = raw_embeddings[
                torch.arange(hidden.size(0), device=hidden.device), position + 1
            ]
            identity_query = F.normalize(self.readout_key(raw_endpoint), dim=-1)
            identity_scores = torch.einsum("bd,btd->bt", identity_query, identity_keys)
            contextual_scores = (
                torch.einsum(
                    "bd,btd->bt",
                    self.readout_query(query + endpoint_identity),
                    self.readout_key(hidden),
                )
                / scale
            )
            scores = identity_scores / 0.10 + contextual_scores
            weights = torch.softmax(scores.masked_fill(~attention_mask.bool(), -1.0e4), dim=-1)
            return torch.einsum("bt,btd->bd", weights, hidden)

        pooled_a = candidate_pool(candidate_a_position)
        pooled_b = candidate_pool(candidate_b_position)
        query_scores = (
            torch.einsum("bd,btd->bt", self.readout_query(query), self.readout_key(hidden)) / scale
        )
        query_weights = torch.softmax(
            query_scores.masked_fill(~attention_mask.bool(), -1.0e4), dim=-1
        )
        pooled_query = torch.einsum("bt,btd->bd", query_weights, hidden)
        score_a = self.answer_head(torch.cat((query, candidate_a, pooled_a), dim=-1))
        score_b = self.answer_head(torch.cat((query, candidate_b, pooled_b), dim=-1))
        abstain = self.abstention_head(torch.cat((query, pooled_query), dim=-1))
        return torch.cat((score_a, score_b, abstain), dim=-1)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._encode(token_ids, attention_mask)
        return self._direct_answer(hidden, token_ids, attention_mask)

    def forward_with_auxiliary(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return training-only predictions without changing the answer path."""

        hidden = self._encode(token_ids, attention_mask)
        pointer_query = self.pointer_query(hidden)
        scale = math.sqrt(float(self.config.hidden_dim))
        source_logits = torch.einsum("btd,bsd->bts", pointer_query, self.source_key(hidden)) / scale
        target_logits = torch.einsum("btd,bsd->bts", pointer_query, self.target_key(hidden)) / scale
        invalid = ~attention_mask.bool()[:, None, :]
        source_logits = source_logits.masked_fill(invalid, -1.0e4)
        target_logits = target_logits.masked_fill(invalid, -1.0e4)
        return {
            "answer_logits": self._direct_answer(hidden, token_ids, attention_mask),
            "entity_logits": self.entity_head(hidden).squeeze(-1),
            "query_role_logits": self.query_role_head(hidden),
            "relation_anchor_logits": self.relation_anchor_head(hidden),
            "event_source_logits": source_logits,
            "event_target_logits": target_logits,
            "boundary_value_logits": self.boundary_value_head(hidden),
            **{
                f"physical_{name}_logits": head(hidden)
                for name, head in self.physical_heads.items()
            },
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def training_only_parameter_count(model: DirectTransformer) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(TRAINING_ONLY_PREFIXES)
    )


def inference_active_parameter_count(model: DirectTransformer) -> int:
    return count_parameters(model) - training_only_parameter_count(model)


def matched_transformer_config(
    *,
    vocab_size: int,
    max_tokens: int,
    query_token_id: int,
    candidate_a_token_id: int,
    candidate_b_token_id: int,
    target_parameters: int,
    candidate_layers: tuple[int, ...] = (4,),
) -> TransformerConfig:
    """Select the closest registered baseline shape to the ERGT parameter count."""

    if not candidate_layers or any(int(layers) <= 0 for layers in candidate_layers):
        raise ValueError("candidate_layers must contain positive layer counts")
    candidates: list[TransformerConfig] = []
    for hidden in (24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128):
        heads = 4 if hidden % 4 == 0 else 2
        # The deployed parameter comparison excludes training-only
        # supervision heads. V6 retains the four-layer default; V7 may
        # preregister another standard depth during baseline qualification.
        for layers in tuple(int(value) for value in candidate_layers):
            for multiplier in (2, 3, 4):
                candidates.append(
                    TransformerConfig(
                        vocab_size=vocab_size,
                        max_tokens=max_tokens,
                        query_token_id=query_token_id,
                        candidate_a_token_id=candidate_a_token_id,
                        candidate_b_token_id=candidate_b_token_id,
                        hidden_dim=hidden,
                        n_heads=heads,
                        n_layers=layers,
                        feedforward_dim=multiplier * hidden,
                    )
                )
    best: TransformerConfig | None = None
    best_delta = float("inf")
    for candidate in candidates:
        model = DirectTransformer(candidate)
        delta = abs(inference_active_parameter_count(model) - target_parameters)
        if delta < best_delta:
            best, best_delta = candidate, delta
    assert best is not None
    return best


def architecture_contract(model: DirectTransformer) -> dict[str, bool]:
    modules = tuple(model.modules())
    return {
        "standard_qkv_self_attention_present": any(
            isinstance(module, nn.MultiheadAttention) for module in modules
        ),
        "direct_answer_head_present": isinstance(model.answer_head, nn.Sequential),
        "external_program_or_solver_absent": model.has_external_program_or_solver is False,
        "raw_token_forward_signature": model.receives_only_raw_tokens is True,
        "training_only_matched_supervision_present": (
            model.uses_training_only_matched_supervision is True
        ),
        "auxiliary_heads_do_not_feed_answer_path": (
            model.auxiliary_heads_feed_answer_path is False
        ),
    }


__all__ = [
    "DirectTransformer",
    "TransformerConfig",
    "architecture_contract",
    "count_parameters",
    "inference_active_parameter_count",
    "matched_transformer_config",
    "training_only_parameter_count",
]
