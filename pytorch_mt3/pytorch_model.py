"""PyTorch implementation of MT3 model for music transcription.

This is a PyTorch re-implementation of the JAX/Flax MT3 model.
While potentially less optimized than the original, it allows for
easier integration with PyTorch-based pipelines.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class T5RMSNorm(nn.Module):
    """T5-style RMS normalization (no mean subtraction, no bias, scale only).

    Matches JAX T5 LayerNorm: y = x * rsqrt(mean(x^2) + eps) * scale
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        mean2 = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(mean2 + self.eps)
        return (self.weight * x).to(input_dtype)


class MT3Config:
    """Configuration for MT3 model.

    Attributes match the JAX implementation's gin config (mt3/gin/model.gin).
    """
    def __init__(
        self,
        vocab_size: int = 1536,  # Will be set from vocabulary
        emb_dim: int = 512,
        num_heads: int = 6,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 8,
        head_dim: int = 64,
        mlp_dim: int = 1024,
        dropout_rate: float = 0.1,
        max_encoder_length: int = 256,  # From mt3.gin TASK_FEATURE_LENGTHS
        max_decoder_length: int = 1024,
        input_depth: int = 512,  # Mel bins
    ):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.input_depth = input_depth


class FixedPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Matches the JAX T5 sinusoidal() initializer exactly:
    - scale_factor = -log(max_scale/min_scale) / (features//2 - 1)
    - div_term = min_scale * exp(arange(features//2) * scale_factor)
    - sin in first half of features, cos in second half
    """
    def __init__(self, emb_dim: int, max_len: int = 5000,
                 min_scale: float = 1.0, max_scale: float = 10000.0):
        super().__init__()
        self.emb_dim = emb_dim

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(max_len).unsqueeze(1).float()

        half_dim = emb_dim // 2
        scale_factor = -math.log(max_scale / min_scale) / (half_dim - 1)
        div_term = min_scale * torch.exp(torch.arange(half_dim).float() * scale_factor)

        pe[:, :half_dim] = torch.sin(position * div_term)
        pe[:, half_dim:2 * half_dim] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, emb_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.

    Matches JAX implementation's MultiHeadDotProductAttention.
    """
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.emb_dim = emb_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(emb_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, num_heads * head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, emb_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, emb_dim]
            key: [batch, seq_len, emb_dim]
            value: [batch, seq_len, emb_dim]
            attention_mask: Optional [batch, 1, seq_len, seq_len] or broadcastable

        Returns:
            Output tensor [batch, seq_len, emb_dim]
        """
        batch_size = query.size(0)

        # Project and reshape to [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [batch, num_heads, seq_len_q, seq_len_k]
        # Note: no explicit 1/sqrt(d) scaling - JAX T5 bakes this into the
        # query weight initialization, so trained weights already include it.
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len_q, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)

        return output


class GatedMLP(nn.Module):
    """Gated MLP with GELU activation.

    Matches JAX implementation with ('gelu', 'linear') activations.
    """
    def __init__(self, emb_dim: int, mlp_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.wi_0 = nn.Linear(emb_dim, mlp_dim, bias=False)  # GELU path
        self.wi_1 = nn.Linear(emb_dim, mlp_dim, bias=False)  # Linear path
        self.wo = nn.Linear(mlp_dim, emb_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, emb_dim]

        Returns:
            Output tensor [batch, seq_len, emb_dim]
        """
        # Gated activation: GELU(x @ W0) * (x @ W1)
        gelu_path = F.gelu(self.wi_0(x))
        linear_path = self.wi_1(x)
        hidden = gelu_path * linear_path
        hidden = self.dropout(hidden)
        output = self.wo(hidden)
        return output


class EncoderLayer(nn.Module):
    """Transformer encoder layer.

    Architecture:
        - Self-attention with pre-LayerNorm
        - Residual connection
        - MLP with pre-LayerNorm
        - Residual connection
    """
    def __init__(self, config: MT3Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            config.emb_dim,
            config.num_heads,
            config.head_dim,
            config.dropout_rate,
        )
        self.mlp = GatedMLP(config.emb_dim, config.mlp_dim, config.dropout_rate)

        self.norm1 = T5RMSNorm(config.emb_dim)
        self.norm2 = T5RMSNorm(config.emb_dim)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, emb_dim]
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, emb_dim]
        """
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, attention_mask)
        x = self.dropout1(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention.

    Architecture:
        - Self-attention with pre-LayerNorm and causal mask
        - Residual connection
        - Cross-attention with pre-LayerNorm
        - Residual connection
        - MLP with pre-LayerNorm
        - Residual connection
    """
    def __init__(self, config: MT3Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            config.emb_dim,
            config.num_heads,
            config.head_dim,
            config.dropout_rate,
        )
        self.cross_attn = MultiHeadAttention(
            config.emb_dim,
            config.num_heads,
            config.head_dim,
            config.dropout_rate,
        )
        self.mlp = GatedMLP(config.emb_dim, config.mlp_dim, config.dropout_rate)

        self.norm1 = T5RMSNorm(config.emb_dim)
        self.norm2 = T5RMSNorm(config.emb_dim)
        self.norm3 = T5RMSNorm(config.emb_dim)

        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.dropout3 = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input [batch, seq_len, emb_dim]
            encoder_output: Encoder output [batch, enc_len, emb_dim]
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Output tensor [batch, seq_len, emb_dim]
        """
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, self_attn_mask)
        x = self.dropout1(x)
        x = residual + x

        # Cross-attention block
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.dropout2(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = self.dropout3(x)
        x = residual + x

        return x


class MT3Encoder(nn.Module):
    """MT3 Encoder: Continuous spectrogram input → contextualized embeddings."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.config = config

        # Project continuous inputs (spectrogram) to embedding dimension
        self.input_projection = nn.Linear(config.input_depth, config.emb_dim, bias=False)

        # Positional encoding
        self.pos_encoding = FixedPositionalEncoding(config.emb_dim, config.max_encoder_length)

        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])

        # Final layer norm
        self.final_norm = T5RMSNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_inputs: Spectrogram features [batch, seq_len, input_depth]
            attention_mask: Optional attention mask

        Returns:
            Encoder output [batch, seq_len, emb_dim]
        """
        # Project continuous inputs to embeddings
        x = self.input_projection(encoder_inputs)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization
        x = self.final_norm(x)
        x = self.dropout(x)

        return x


class MT3Decoder(nn.Module):
    """MT3 Decoder: Token input + encoder output → output logits."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.emb_dim)

        # Positional encoding
        self.pos_encoding = FixedPositionalEncoding(config.emb_dim, config.max_decoder_length)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])

        # Final layer norm
        self.final_norm = T5RMSNorm(config.emb_dim)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_input_ids: Input token IDs [batch, seq_len]
            encoder_output: Encoder output [batch, enc_len, emb_dim]
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.token_embeddings(decoder_input_ids)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        # Final normalization
        x = self.final_norm(x)
        x = self.dropout(x)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits


class MT3Model(nn.Module):
    """Complete MT3 model: Encoder-Decoder transformer for music transcription."""

    def __init__(self, config: MT3Config):
        super().__init__()
        self.config = config
        self.encoder = MT3Encoder(config)
        self.decoder = MT3Decoder(config)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_inputs: Spectrogram features [batch, enc_len, input_depth]
            decoder_input_ids: Decoder input tokens [batch, dec_len]
            decoder_attention_mask: Optional attention mask for decoder

        Returns:
            Logits [batch, dec_len, vocab_size]
        """
        # Encode
        encoder_output = self.encoder(encoder_inputs)

        # Create causal mask for decoder self-attention
        if decoder_attention_mask is None:
            seq_len = decoder_input_ids.size(1)
            decoder_attention_mask = self._create_causal_mask(seq_len, decoder_input_ids.device)

        # Decode
        logits = self.decoder(
            decoder_input_ids,
            encoder_output,
            self_attn_mask=decoder_attention_mask,
        )

        return logits

    @staticmethod
    def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation.

        Returns:
            Mask [1, 1, seq_len, seq_len] where future positions are -inf
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        encoder_inputs: torch.Tensor,
        max_length: int = 1024,
        start_token_id: int = 0,
        eos_token_id: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation using greedy decoding.

        Args:
            encoder_inputs: Spectrogram features [batch, enc_len, input_depth]
            max_length: Maximum sequence length to generate
            start_token_id: ID of start token
            eos_token_id: ID of end-of-sequence token
            temperature: Sampling temperature (1.0 = no change)

        Returns:
            Generated token IDs [batch, seq_len]
        """
        batch_size = encoder_inputs.size(0)
        device = encoder_inputs.device

        # Encode once
        encoder_output = self.encoder(encoder_inputs)

        # Start with start token
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            # Create causal mask
            seq_len = generated.size(1)
            causal_mask = self._create_causal_mask(seq_len, device)

            # Get logits for current sequence
            logits = self.decoder(generated, encoder_output, self_attn_mask=causal_mask)

            # Get logits for last position and apply temperature
            next_token_logits = logits[:, -1, :] / temperature

            # Greedy sampling
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have generated EOS
            if (next_token == eos_token_id).all():
                break

        return generated
