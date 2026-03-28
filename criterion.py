import segmentation_models_pytorch_3d as smp
from torch import nn


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal_loss = smp.losses.FocalLoss(mode="binary", alpha=0.25)
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, inputs, targets):
        focal_loss_value = self.focal_loss(inputs, targets)
        dice_loss_value = self.dice_loss(inputs, targets)

        return self.focal_weight * focal_loss_value + self.dice_weight * dice_loss_value
