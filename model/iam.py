import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = (dim / num_heads) ** (-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def _apply_position(self, tensor, position):
        if position is None:
            return tensor
        if tensor.ndim != position.ndim:
            tensor = rearrange(
                tensor, "b n (g c) -> b n g c", g=self.num_heads
            )
            tensor = tensor + position
            return rearrange(tensor, "b n g c -> b n (g c)")
        return tensor + position

    def forward(self, query, key, value, query_position=None, key_position=None):
        query = self.norm_q(self._apply_position(query, query_position))
        key = self.norm_k(self._apply_position(key, key_position))
        value = self.norm_v(value)
        query = self.to_q(query)
        key = self.to_k(key)
        value = self.to_v(value)

        query = rearrange(
            query, "b n (g c) -> b n g c", g=self.num_heads
        )
        key = rearrange(key, "b n (g c) -> b n g c", g=self.num_heads)
        value = rearrange(
            value, "b n (g c) -> b n g c", g=self.num_heads
        )

        attention = torch.einsum("b q g c, b k g c -> b q g k", query, key)
        attention = (attention * self.scale).softmax(dim=-1)
        output = torch.einsum(
            "b q g k, b k g c -> b q g c", attention, value.float()
        )
        output = rearrange(output, "b q g c -> b q (g c)")
        return self.proj(output)


class IAM(nn.Module):
    """Invariant Affordance Knowledge Extraction Module."""

    def __init__(
        self,
        dim=384,
        num_heads=6,
        qkv_bias=False,
        invariant_extract_layers=5,
        image_count=2,
        image_token_count=49,
        drop_rate=0.0,
    ):
        super().__init__()
        self.query_cross_attention = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MultiHeadAttention(dim, num_heads, qkv_bias)
                        for _ in range(invariant_extract_layers)
                    ]
                )
                for _ in range(image_count)
            ]
        )
        self.num_layers = invariant_extract_layers
        self.image_self_attention = nn.ModuleList(
            [
                MultiHeadAttention(dim, num_heads, qkv_bias)
                for _ in range(invariant_extract_layers)
            ]
        )
        self.image_query_cross_attention = nn.ModuleList(
            [
                MultiHeadAttention(dim, num_heads, qkv_bias)
                for _ in range(invariant_extract_layers)
            ]
        )
        self.query_aggregation = nn.ModuleList(
            [
                nn.Linear(image_count * dim, dim)
                for _ in range(invariant_extract_layers)
            ]
        )
        self.dropout = nn.Dropout(drop_rate)
        self.image_token_count = image_token_count
        self.dim = dim
        self.image_count = image_count
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.classifier_pool = nn.AdaptiveAvgPool1d(1)
        self.invariant_feature_projection = nn.Linear(image_count * dim, dim)
        self.affordance_classifier = nn.Sequential(
            nn.Linear(image_count * dim, dim // image_count),
            nn.BatchNorm1d(dim // image_count),
            nn.ReLU(),
            nn.Linear(dim // image_count, 17),
            nn.BatchNorm1d(17),
        )

    def forward(self, image_features):
        initial_queries = [
            torch.zeros(
                1,
                self.image_token_count,
                self.dim,
                device=image_features[0].device,
            )
            for _ in range(self.image_count)
        ]
        query_layers = []
        image_feature_layers = []
        similarity_loss = 0

        for layer_index in range(self.num_layers):
            query_layers.append([])
            image_feature_layers.append([])

            for image_index, features in enumerate(image_features):
                query = (
                    initial_queries[image_index]
                    if layer_index == 0
                    else query_layers[-2][image_index]
                )
                query_layers[-1].append(
                    self.query_cross_attention[image_index][layer_index](
                        query, features, features
                    )
                )

                self_attention_input = (
                    features
                    if layer_index == 0
                    else image_feature_layers[-2][image_index]
                )
                image_feature_layers[-1].append(
                    self.image_self_attention[layer_index](
                        self_attention_input,
                        self_attention_input,
                        self_attention_input,
                    )
                )

            aggregated_queries = torch.cat(query_layers[-1], dim=2)
            aggregated_queries = self.dropout(
                self.query_aggregation[layer_index](aggregated_queries)
            )

            for image_index in range(self.image_count):
                features = image_feature_layers[-1][image_index]
                image_feature_layers[-1][image_index] = (
                    self.image_query_cross_attention[layer_index](
                        aggregated_queries, features, features
                    )
                )

            for first_index in range(self.image_count - 1):
                for second_index in range(first_index + 1, self.image_count):
                    first = image_feature_layers[-1][first_index].flatten(1)
                    second = image_feature_layers[-1][second_index].flatten(1)
                    similarity_loss = similarity_loss + (
                        1 - self.cosine_similarity(first, second)
                    )

        image_queries = torch.stack(query_layers[-1], dim=0)
        concatenated_features = torch.cat(image_feature_layers[-1], dim=2)
        pooled_features = rearrange(concatenated_features, "b n d -> b d n")
        logits = self.affordance_classifier(
            self.classifier_pool(pooled_features).flatten(1)
        )
        invariant_features = self.invariant_feature_projection(
            concatenated_features
        )
        return {
            "similarity_loss": similarity_loss,
            "logits": logits,
            "invariant_features": invariant_features,
            "image_queries": image_queries,
        }
