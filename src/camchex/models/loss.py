import torch
import torch.nn as nn

class ASL(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(ASL, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        # Debug: check for NaNs in inputs
        if torch.isnan(pred).any():
            print("NaN detected in `pred`")
        if torch.isnan(label).any():
            print("NaN detected in `label`")

        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()

        xs_pos = torch.sigmoid(pred)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(min=self.eps, max=1 - self.eps)

        xs_pos = xs_pos.clamp(min=self.eps, max=1 - self.eps)
        xs_neg = xs_neg.clamp(min=self.eps, max=1 - self.eps)

        los_pos = label * torch.log(xs_pos)
        los_neg = (1 - label) * torch.log(xs_neg)
        loss = los_pos + los_neg
        loss *= weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label
            pt1 = xs_neg * (1 - label)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        # Final loss
        final_loss = -loss.mean()

        return final_loss