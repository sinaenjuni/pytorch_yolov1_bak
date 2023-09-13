import torch

def intersection_over_union(box_preds, box_targets):
    # minx, miny, maxx, maxy    
    minx = torch.max(box_preds[..., 0], box_targets[..., 0])
    miny = torch.max(box_preds[..., 1], box_targets[..., 1])
    maxx = torch.min(box_preds[..., 2], box_targets[..., 2])
    maxy = torch.min(box_preds[..., 3], box_targets[..., 3])
    intersection = (maxx - minx).clamp(0) * (maxy - miny).clamp(0)
    # print(intersection)

    # w: maxx - minx, h:maxy - miny
    preds_area =   torch.abs((box_preds[..., 2] - box_preds[..., 0]) * (box_preds[..., 3] - box_preds[..., 1]))
    targets_area = torch.abs((box_targets[..., 2] - box_targets[..., 0]) * (box_targets[..., 3] - box_targets[..., 1]))
    union = (preds_area + targets_area - intersection + 1e-6)
    # print(union)
    return intersection / union

if __name__ == "__main__":
    N = 32
    box_preds = torch.randint(0, 640, (N, 4))
    box_targets = torch.randint(0, 640, (N, 4))
    intersection_over_union(box_preds, box_targets)
