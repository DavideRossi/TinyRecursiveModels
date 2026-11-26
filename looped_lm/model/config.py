"""Model configuration: every hyperparameter of the model family, saved as JSON with each run."""
import dataclasses
import json
import math

from ..dataset_generation.vocab import VOCAB


@dataclasses.dataclass
class ModelConfig:
    """embedding -> prelude (P layers) -> core (C layers) x n_loops -> coda (K layers) -> RMSNorm -> head

    width: residual stream size d; num_heads x head_dim: attention size (= width by default)
    mlp_hidden: SwiGLU hidden size m; init_std: std of every weight at initialization except the
    embedding (the output projections of attention and MLP get init_std / sqrt(2 * num_layers));
    embed_init_std: std of the (tied) embedding, 0 for 1 / sqrt(width), resolved at construction
    so that the saved config holds the value used."""
    vocab_size: int = VOCAB.size
    width: int = 256
    num_heads: int = 4
    head_dim: int = 64
    mlp_hidden: int = 704
    prelude_layers: int = 0
    core_layers: int = 8
    coda_layers: int = 0
    n_loops: int = 1
    max_positions: int = 8192
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    embed_init_std: float = 0.0

    def __post_init__(self):
        if self.head_dim % 2:
            raise ValueError("head_dim must be even (rotary embeddings)")
        for k in ("vocab_size", "width", "num_heads", "head_dim", "mlp_hidden", "max_positions"):
            if getattr(self, k) < 1:
                raise ValueError(f"{k} must be >= 1")
        if min(self.prelude_layers, self.core_layers, self.coda_layers) < 0 or self.num_layers < 1:
            raise ValueError("layer counts must be >= 0, with at least one layer")
        if self.embed_init_std < 0:
            raise ValueError("embed_init_std must be >= 0 (0: 1 / sqrt(width))")
        if self.embed_init_std == 0:
            self.embed_init_std = 1.0 / math.sqrt(self.width)

    @property
    def num_layers(self) -> int:
        """Distinct layers (P + C + K): the depth used to scale the output projections."""
        return self.prelude_layers + self.core_layers + self.coda_layers

    def expected_num_params(self) -> int:
        """V*d (tied embedding) + d (final norm) + per layer: q, k, v, o + SwiGLU + 2 norm gains."""
        d, a, m = self.width, self.num_heads * self.head_dim, self.mlp_hidden
        per_layer = 4 * d * a + 3 * d * m + 2 * d
        return self.vocab_size * d + d + self.num_layers * per_layer

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)
