import torch
import torch.nn as nn


class HM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, prediction, target):
        positive = -(1 - self.alpha) * prediction**self.gamma
        positive = positive * (1 - target) * torch.log(1 - prediction + 1e-6)

        negative = -self.alpha * (1 - prediction) ** self.gamma
        negative = negative * target * torch.log(prediction + 1e-6)
        ce_loss = torch.sum(torch.mean(positive + negative, (0, 1)))

        intersection_positive = torch.sum(prediction * target, 1)
        cardinality_positive = torch.sum(
            torch.abs(prediction) + torch.abs(target), 1
        )
        dice_positive = (intersection_positive + 1e-6) / (
            cardinality_positive + 1e-6
        )

        intersection_negative = torch.sum((1 - prediction) * (1 - target), 1)
        cardinality_negative = torch.sum(
            2 - torch.abs(prediction) - torch.abs(target), 1
        )
        dice_negative = (intersection_negative + 1e-6) / (
            cardinality_negative + 1e-6
        )
        dice_loss = torch.sum(torch.mean(1.5 - dice_positive - dice_negative, 0))
        return ce_loss + dice_loss
