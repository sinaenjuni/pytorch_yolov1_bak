import torch

def intersection_over_union(pred_bboxes, target_bboxes):
    # minx, miny, maxx, maxy    
    minx = torch.max(pred_bboxes[..., 0], target_bboxes[..., 0])
    miny = torch.max(pred_bboxes[..., 1], target_bboxes[..., 1])
    maxx = torch.min(pred_bboxes[..., 2], target_bboxes[..., 2])
    maxy = torch.min(pred_bboxes[..., 3], target_bboxes[..., 3])
    intersection = (maxx - minx).clamp(0) * (maxy - miny).clamp(0)
    # print(intersection)

    # w: maxx - minx, h:maxy - miny
    preds_area =   torch.abs((pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1]))
    targets_area = torch.abs((target_bboxes[..., 2] - target_bboxes[..., 0]) * (target_bboxes[..., 3] - target_bboxes[..., 1]))
    union = (preds_area + targets_area - intersection + 1e-6)
    # print(union)
    return intersection / union

if __name__ == "__main__":
    pred_bboxes = torch.tensor([
        [20, 20, 50, 50],
        [0, 0, 25, 25],
        [0, 0, 50, 50],
        ])
    target_bboxes = torch.tensor([
        [20, 20, 50, 50],
        [20, 20, 50, 50],
        [25, 25, 50, 50],
        ])

    iou = intersection_over_union(pred_bboxes, target_bboxes)
    print(iou)