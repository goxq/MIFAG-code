import torch
import torch.nn as nn
from einops import rearrange
from torchvision import models

from model.adm import ADM
from model.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstractionMsg


class PointEncoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, n_p):
        super().__init__()
        self.N_p = n_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(
            512,
            [0.1, 0.2, 0.4],
            [32, 64, 128],
            3 + additional_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128,
            [0.4, 0.8],
            [64, 128],
            128 + 128 + 64,
            [[128, 128, 256], [128, 196, 256]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            self.N_p,
            [0.2, 0.4],
            [16, 32],
            256 + 256,
            [[128, 128, 256], [128, 196, 256]],
        )

    def forward(self, xyz):
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return [
            [l0_xyz, l0_points],
            [l1_xyz, l1_points],
            [l2_xyz, l2_points],
            [l3_xyz, l3_points],
        ]


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        return self.model.layer4(out)


class AffordanceDecoder(nn.Module):
    def __init__(self, additional_channel, emb_dim, n_p, n_raw, expanded=True):
        class SwapAxes(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.emb_dim = emb_dim
        self.N_p = n_p
        self.N = n_raw

        if expanded:
            # Expanded decoder used by the released checkpoints.
            self.fp3 = PointNetFeaturePropagation(
                in_channel=512 + self.emb_dim, mlp=[2048, 1024, 768, 512]
            )
            self.fp2 = PointNetFeaturePropagation(
                in_channel=832, mlp=[2048, 1024, 768, 512]
            )
            self.fp1 = PointNetFeaturePropagation(
                in_channel=518 + additional_channel, mlp=[2048, 1024, 512, 512]
            )
        else:
            # The no-IAM/no-ADM baseline from Table 2.
            self.fp3 = PointNetFeaturePropagation(
                in_channel=512 + self.emb_dim, mlp=[768, 512]
            )
            self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512])
            self.fp1 = PointNetFeaturePropagation(
                in_channel=518 + additional_channel, mlp=[512, 512]
            )

        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim // 8, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_feature):
        p_0, p_1, p_2, p_3 = point_feature
        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], p_3[1])
        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)
        up_sample = self.fp1(
            p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]], 1), up_sample
        )
        return self.sigmoid(self.out_head(up_sample.mT))


class MIFAG(nn.Module):
    """MIFAG and the ablations reported in Tables 2-4 of the paper."""

    VALID_VARIANTS = {"full", "baseline", "iam_only", "adm_only"}

    def __init__(
        self,
        img_model_path=None,
        pre_train=True,
        normal_channel=False,
        local_rank=None,
        N_p=64,
        emb_dim=512,
        N_raw=2048,
        args=None,
    ):
        super().__init__()
        del local_rank

        self.args = args
        self.variant = args.variant
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown variant '{self.variant}'. Choose from {sorted(self.VALID_VARIANTS)}."
            )

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.normal_channel = normal_channel
        self.additional_channel = 3 if normal_channel else 0
        self.image_count = args.train_img_nums

        self.image_encoder = ImageEncoder()
        if pre_train:
            if not img_model_path:
                raise ValueError("img_model_path is required when pre_train=True")
            pretrain_dict = torch.load(img_model_path, map_location="cpu")
            img_model_dict = self.image_encoder.state_dict()
            pretrain_dict = {
                f"model.{key}": value
                for key, value in pretrain_dict.items()
                if f"model.{key}" in img_model_dict
            }
            img_model_dict.update(pretrain_dict)
            self.image_encoder.load_state_dict(img_model_dict)

        self.point_encoder = PointEncoder(
            self.emb_dim, self.normal_channel, self.additional_channel, self.N_p
        )

        if self.variant == "adm_only":
            self._build_adm_only()
        elif self.variant in {"full", "iam_only"}:
            self._build_with_iam()
        else:
            self._build_baseline()

    def _build_adm_only(self):
        import model.vision_transformer as vits

        drop_rate = self.args.droprate
        fuser_kwargs = self._fuser_dropout_kwargs(drop_rate)
        self.point_affordance_fuser = vits.vit_small(
            img_num_patches=64 * self.image_count,
            point_num_patches=64,
            img_numbers=1,
            use_cross_attn=True,
            **fuser_kwargs,
        )
        self._build_adm(drop_rate)
        self.image_token_projection = nn.Linear(512, 384)
        self.point_token_projection = nn.Linear(512, 384)
        self.point_feature_projection = nn.Linear(384, 512)
        self.affordance_decoder = AffordanceDecoder(
            self.additional_channel, self.emb_dim, self.N_p, self.N_raw, expanded=True
        )

    def _build_with_iam(self):
        import model.vision_transformer as vits
        from model.iam import IAM

        self.iam = IAM(
            dim=384,
            num_heads=6,
            qkv_bias=True,
            invariant_extract_layers=self.args.invariant_extract_layers,
            image_count=self.image_count,
            image_token_count=49,
            drop_rate=self.args.droprate,
        )
        self.iam_image_projection = nn.ModuleList(
            [nn.Linear(512, 384) for _ in range(self.image_count)]
        )
        self.point_token_projection = nn.Linear(512, 384)

        if self.variant == "full":
            self.point_affordance_fuser = vits.vit_small(
                img_num_patches=64 * self.image_count,
                point_num_patches=64,
                img_numbers=1,
                use_cross_attn=True,
                **self._fuser_dropout_kwargs(self.args.droprate),
            )
            self._build_adm(self.args.droprate)
        else:
            self.iam_point_fuser = vits.vit_small(
                img_num_patches=49,
                point_num_patches=64,
                img_numbers=1,
                use_cross_attn=True,
            )

        self.point_feature_projection = nn.Linear(384, 512)
        self.affordance_decoder = AffordanceDecoder(
            self.additional_channel, self.emb_dim, self.N_p, self.N_raw, expanded=True
        )

    def _build_baseline(self):
        self.baseline_image_projection = nn.Linear(49, 64)
        self.affordance_decoder = AffordanceDecoder(
            self.additional_channel, self.emb_dim, self.N_p, self.N_raw, expanded=False
        )
        self.baseline_fusion = nn.Conv2d(128, 64, kernel_size=1)

    def _build_adm(self, drop_rate):
        attention_drop = drop_rate if drop_rate else 0.1
        self.adm = ADM(dim=384, attention_drop=attention_drop)

    @staticmethod
    def _fuser_dropout_kwargs(drop_rate):
        if not drop_rate:
            return {}
        return {
            "drop_rate": drop_rate,
            "attn_drop_rate": drop_rate,
            "drop_path_rate": drop_rate,
        }

    def forward(self, images, xyz):
        if len(images) != self.image_count:
            raise ValueError(
                f"Expected {self.image_count} reference images, received {len(images)}."
            )

        point_features = self.point_encoder(xyz)
        image_features = [self.image_encoder(image) for image in images]

        if self.variant == "adm_only":
            return self._forward_adm_only(image_features, point_features)
        if self.variant in {"full", "iam_only"}:
            return self._forward_with_iam(image_features, point_features)
        return self._forward_baseline(image_features, point_features)

    def _forward_adm_only(self, image_features, point_features):
        image_query = rearrange(
            torch.stack(image_features, dim=0), "i b c h w -> i b (h w) c"
        )
        image_query = self.image_token_projection(image_query)
        point = self._point_tokens(point_features)
        weighted_fusion = self.adm(image_query, point)
        return self._decode_fusion(weighted_fusion, point_features)

    def _forward_with_iam(self, image_features, point_features):
        transformed_images = [
            self.iam_image_projection[index](
                rearrange(feat, "b c h w -> b (h w) c")
            )
            for index, feat in enumerate(image_features)
        ]
        iam_output = self.iam(transformed_images)
        point = self._point_tokens(point_features)

        if self.variant == "full":
            weighted_fusion = self.adm(iam_output["image_queries"], point)
            prediction = self._decode_fusion(weighted_fusion, point_features)
        else:
            invariant_features = rearrange(
                iam_output["invariant_features"], "b n c -> b c n"
            )
            fusion = self.iam_point_fuser([invariant_features], point)
            prediction = self._decode_point_tokens(
                fusion[:, -self.N_p :, :], point_features
            )

        return prediction, iam_output["similarity_loss"], iam_output["logits"]

    def _forward_baseline(self, image_features, point_features):
        image = sum(image_features)
        image = image.view(image.shape[0], self.emb_dim, -1).permute(0, 2, 1)
        image = self.baseline_image_projection(image.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        point = point_features[-1][1].permute(0, 2, 1)
        fused = self.baseline_fusion(
            torch.cat([point, image], dim=1).unsqueeze(-1)
        ).squeeze(-1)
        point_features[-1][1] = fused.permute(0, 2, 1)
        return self.affordance_decoder(point_features)

    def _point_tokens(self, point_features):
        point = point_features[-1][1].permute(0, 2, 1)
        return self.point_token_projection(point).permute(0, 2, 1)

    def _decode_fusion(self, weighted_fusion, point_features):
        point = self._point_tokens(point_features)
        fusion = self.point_affordance_fuser([weighted_fusion], point)
        return self._decode_point_tokens(fusion[:, -self.N_p :, :], point_features)

    def _decode_point_tokens(self, point_tokens, point_features):
        point_tokens = self.point_feature_projection(point_tokens)
        point_features[-1][1] = point_tokens.permute(0, 2, 1)
        return self.affordance_decoder(point_features)


def get_MIFAG(
    img_model_path=None,
    pre_train=True,
    normal_channel=False,
    local_rank=None,
    N_p=64,
    emb_dim=512,
    N_raw=2048,
    args=None,
):
    return MIFAG(
        img_model_path,
        pre_train,
        normal_channel,
        local_rank,
        N_p,
        emb_dim,
        N_raw,
        args,
    )
