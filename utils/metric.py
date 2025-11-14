import copy
import numpy as np

def metric(gt, pred):
    preds = pred.detach().cpu().numpy()
    gts = gt.detach().cpu().numpy()

    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)

    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)

    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)

    # 像素级交并
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    # TP / FP / FN / TN
    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape, dtype=int) - union

    tp = np.sum(tp_array)
    fp = np.sum(fp_array)
    fn = np.sum(fn_array)
    tn = np.sum(tn_array)

    smooth = 1e-3

    # 原有指标
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)

    FPR = fp / (fp + tn + smooth)
    FNR = fn / (fn + tp + smooth)

    iou = intersection_sum / (union_sum + smooth)
    dice = 2 * intersection_sum / (gdth_sum + pred_sum + smooth)

    # ===== 新增指标 =====
    # 1) Overall Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn + smooth)

    # 2) Specificity / TNR
    specificity = tn / (tn + fp + smooth)

    # 3) NPV (Negative Predictive Value)
    npv = tn / (tn + fn + smooth)

    # 4) F1-score
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)

    # 5) Balanced Accuracy
    balanced_acc = (recall + specificity) / 2.0

    # 6) MCC
    mcc_den = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + smooth
    )
    mcc = (tp * tn - fp * fn) / mcc_den

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "FPR": FPR,
        "FNR": FNR,
        "acc": acc,
        "specificity": specificity,
        "NPV": npv,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "MCC": mcc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
