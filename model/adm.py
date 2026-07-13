import torch
import torch.nn as nn


class IQDCA(nn.Module):
    """Invariant-aware Query Dictionary Cross-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, dictionary):
        batch_size, query_count, channels = query.shape
        dictionary_count = dictionary.shape[1]
        query = self.q(query).reshape(
            batch_size,
            query_count,
            self.num_heads,
            channels // self.num_heads,
        )
        query = query.permute(0, 2, 1, 3)
        key_value = self.kv(dictionary).reshape(
            batch_size,
            dictionary_count,
            2,
            self.num_heads,
            channels // self.num_heads,
        )
        key_value = key_value.permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(
            batch_size, query_count, channels
        )
        output = self.proj_drop(self.proj(output))
        return output, attention


class SWA(nn.Module):
    """Self-Weighted Attention for affordance dictionary fusion."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, features):
        batch_size, token_count, channels = features.shape
        query_key_value = self.qkv(features).reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value[0], query_key_value[1], query_key_value[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(
            batch_size, token_count, channels
        )
        output = self.proj_drop(self.proj(output))
        return output, attention


class ADM(nn.Module):
    """Affordance Dictionary Adaptive Fusion Module."""

    def __init__(self, dim=384, attention_drop=0.1):
        super().__init__()
        self.iqdca = IQDCA(
            dim=dim,
            attn_drop=attention_drop,
            proj_drop=attention_drop,
        )
        self.swa = SWA(
            dim=dim,
            attn_drop=attention_drop,
            proj_drop=attention_drop,
        )

    def forward(self, image_queries, point_tokens):
        point_queries = point_tokens.permute(0, 2, 1)
        dictionary_fusions = [
            self.iqdca(point_queries, image_query)[0]
            for image_query in image_queries
        ]
        dictionary_fusions = torch.stack(dictionary_fusions, dim=0).permute(
            1, 2, 0, 3
        )
        dictionary_fusions = dictionary_fusions.reshape(
            dictionary_fusions.shape[0],
            -1,
            dictionary_fusions.shape[-1],
        )
        return self.swa(dictionary_fusions)[0].permute(0, 2, 1)
