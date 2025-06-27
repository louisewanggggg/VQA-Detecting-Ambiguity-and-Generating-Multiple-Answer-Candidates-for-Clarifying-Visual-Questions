from io import open
import json
import logging
import os
import sys
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import DatasetMapTrain, DatasetMapEval
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb
import numpy as np

logger = logging.getLogger(__name__)

# image_id_to_objects = torch.load("/root/simmc_2/simmc2/resnet_features_objectlevel.pt")
LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}

TP = FP = FN = TN = 0
def compute_iou(box1, box2):
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().tolist()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().tolist()
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_center_distance(box1, box2):
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().tolist()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().tolist()
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def compute_area_ratio(box1, box2):
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().tolist()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().tolist()
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return min(area1, area2) / (max(area1, area2) + 1e-6)


def objects_spatially_ambiguous(bboxes, matched_indices, iou_threshold=0.5, dist_threshold=0.2, area_threshold=0.6):
    if len(matched_indices) <= 1:
        return False
    for i in range(len(matched_indices)):
        for j in range(i + 1, len(matched_indices)):
            box1, box2 = bboxes[matched_indices[i]], bboxes[matched_indices[j]]
            iou = compute_iou(box1, box2)
            dist = compute_center_distance(box1, box2)
            area_ratio = compute_area_ratio(box1, box2)
            if iou > iou_threshold or dist < dist_threshold or area_ratio > area_threshold:
                return True
    return False


def mentions_multiple_objects(question):
    multi_target_keywords = [
        "both", "these", "those", "all the", "three", "two", "several",
        "multiple", "identical", "and", "a few", "many"
    ]
    question_lower = question.lower()
    return any(k in question_lower for k in multi_target_keywords)

def is_ambiguous_full(
        question, object_scores, bboxes, entropy, num_answers, pred_answer_count,
        score_threshold=0.2, diff_margin=0.1, entropy_threshold=1.7
):
    matched = object_scores > score_threshold
    matched_indices = torch.nonzero(matched).squeeze(-1)
    matched_scores = object_scores[matched]

    if len(matched_indices) == 0:
        spatial_amb = False
        score_diff = False
        exp_entropy = False
        language_guided = mentions_multiple_objects(question)  # 这里定义一下
        ambiguous = True
        entropy_alignment_loss = abs(num_answers - 1.0)

    elif len(matched_indices) == 1:
        spatial_amb = False
        score_diff = False
        exp_entropy = False
        language_guided = mentions_multiple_objects(question)  # 同样定义一下
        ambiguous = False
        entropy_alignment_loss = abs(num_answers - 1.0)

    else:
        top_score = matched_scores.max()
        score_diff = top_score - matched_scores.mean()
        spatial_amb = objects_spatially_ambiguous(bboxes, matched_indices)
        exp_entropy = torch.exp(entropy).item() if isinstance(entropy, torch.Tensor) else np.exp(entropy)
        language_guided = mentions_multiple_objects(question)
        entropy_alignment_loss = abs(num_answers - exp_entropy)

        score = 0
        if entropy_alignment_loss > 1.0:
            score += 1
        if spatial_amb:
            score += 1
        if score_diff < diff_margin:
            score += 1
        if not language_guided and pred_answer_count >= 2:
            score += 1
        if not language_guided and exp_entropy > entropy_threshold:
            score += 1

        ambiguous = score >= 2

    return (
        entropy_alignment_loss,
        spatial_amb,
        score_diff,
        exp_entropy,
        ambiguous,
        language_guided,
    )

def compute_loss_with_entropy(cls_loss, entropy_alignment_loss, ambiguity_flag, is_ambiguous_gt, ambiguous_logit=None,
                              lambda_entropy=1.5, lambda_ambiguity=0.5):
    # ambiguity classification loss from ambiguous_logit (if given)
    if ambiguous_logit is not None:
        ambiguity_gt = torch.tensor([float(is_ambiguous_gt)]).to(cls_loss.device)
        ambiguity_loss = F.binary_cross_entropy_with_logits(ambiguous_logit.view(-1), ambiguity_gt)
        return cls_loss + lambda_entropy * entropy_alignment_loss + lambda_ambiguity * ambiguity_loss
    else:
        ambiguity_pred = torch.tensor([float(ambiguity_flag)]).to(cls_loss.device)
        ambiguity_gt = torch.tensor([float(is_ambiguous_gt)]).to(cls_loss.device)
        ambiguity_loss = F.binary_cross_entropy(ambiguity_pred, ambiguity_gt)
        return cls_loss + lambda_entropy * entropy_alignment_loss + lambda_ambiguity * ambiguity_loss


def update_metrics(results, predicted, groundtruth):
    global TP, FP, FN, TN
    if predicted and groundtruth:
        TP += 1
    elif predicted and not groundtruth:
        FP += 1
    elif not predicted and groundtruth:
        FN += 1
    else:
        TN += 1

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    if isinstance(results, list):
        if isinstance(results[0], dict):
            results[0]["precision"] = precision
            results[0]["recall"] = recall
            results[0]["f1"] = f1
    elif isinstance(results, dict):
        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = f1

    return results


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses, threshold):
    # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    batch = tuple(t.cuda(device=device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
    # weights = load_label_weights("/root/vilbert-multi-task/datasets/VQA/cache/label_frequencies.pkl")
    # weights = weights.to(device)

    if task_id == "TASK4" or task_id == "TASK17":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, image_id, is_ambiguous = (
            batch
        )

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    text_emb, image_emb, vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, ambiguous_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        cls_loss = loss_fn(vil_prediction, target)
        cls_loss = cls_loss.mean() * target.size(1)
        # loss = loss.mean() * target.size(1)
        # batch_score = compute_score_with_logits(vil_prediction, target).sum()
        # target = (target >= threshold).float()
        true_labels_count = target.sum(dim=1)
        preds = torch.sigmoid(vil_prediction)
        preds_bin = (preds >= threshold).float()
        sample_scores = torch.zeros(preds.size(0)).to(preds.device)
        k = 3
        min_threshold = 0.2
        entropy_list = []
        results = []
        image_id_to_objects = torch.load("/root/simmc_2/simmc2/resnet_features_objectlevel.pt")
        ambiguous_labels = batch[10].float().to(preds.device)
        for i in range(preds.size(0)):  # 遍历每个样本
            sigmoid_vil = torch.sigmoid(vil_prediction)
            predicted_labels = preds_bin[i]  # 当前样本的预测标签
            # current_true_label_count = int(true_labels_count[i].item())  # 当前样本真实标签的数量
            if predicted_labels.sum() == 0:  # 如果预测标签数量少于真实标签数量
                adjusted_preds = (preds[i] >= min_threshold).float()
                if adjusted_preds.sum() > 0:
                    dynamic_k = min(k, int(adjusted_preds.sum().item()))
                    top_k_indices = torch.topk(adjusted_preds, dynamic_k).indices
                    adjusted_preds = torch.zeros_like(adjusted_preds)  # 初始化为全 0
                    adjusted_preds[top_k_indices] = 1.0  # 只保留 top-k 标签

                preds_bin[i] = adjusted_preds
            intersection = (preds_bin[i] * target[i]).sum().float()  # 交集 (True Positives)
            union = (preds_bin[i] + target[i]).clamp(0, 1).sum().float()  # 并集
            sample_scores[i] = intersection / (union + 1e-6)
            pred_i = preds_bin[i]
            filtered_sigmoid = pred_i[pred_i >= threshold]
            if filtered_sigmoid.numel() == 0:
                k = 10
                topk_vals = torch.topk(pred_i, k=min(k, pred_i.size(0))).values
                norm_probs = topk_vals / (topk_vals.sum() + 1e-12)
            else:
                norm_probs = filtered_sigmoid / (filtered_sigmoid.sum() + 1e-12)

            entropy_i = -(norm_probs * (norm_probs + 1e-12).log()).sum()
            avg_object_sim = -1.0

            if image_id_to_objects is not None:
                image_key = str(image_id[i])
                object_feats_dict = image_id_to_objects.get(image_key)
                if object_feats_dict:
                    obj_feats = torch.stack([v['feature'].to(text_emb.device) for v in object_feats_dict.values()])
                    obj_feats = F.normalize(obj_feats, dim=-1)
                    text_feat = F.normalize(text_emb[i], dim=-1)
                    text_feat_proj = model.module.text_proj(text_feat)
                    sim_scores = torch.matmul(obj_feats, F.normalize(text_feat_proj, dim=-1))
                    sim_scores_list = sim_scores.tolist()
                    top_sim = sim_scores.topk(5).values
                    avg_object_sim = top_sim.mean().item()
            results.append({
                "question": question_id[i],

                "vil_sigmoid": preds[i],
                "preds": preds_bin[i],

                "target": target[i],
                "sim_scores": sim_scores_list,
                "avg_object_sim": avg_object_sim,
                "ambiguous_labels": ambiguous_labels[i].item(),

            })
        entropy_tensor = torch.stack(entropy_list)
        entropy_logit = entropy_tensor.unsqueeze(1)  # [B, 1]
        entropy_pred = model.module.entropy_classifier(entropy_logit).squeeze(1)
        entropy_loss = torch.nn.BCEWithLogitsLoss()(entropy_pred, ambiguous_labels)
        lambda_entropy = 0.5
        loss = cls_loss + lambda_entropy * entropy_loss
        entropy_acc = ((torch.sigmoid(entropy_pred) >= 1.0).float() == ambiguous_labels).float().mean()

    if task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = task_losses[task_id](vil_prediction_gqa, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(vil_prediction_gqa, target).sum()

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = (preds == target).sum()

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return loss, batch_size, entropy_acc, results


def ForwardModelsTrain(
        args,
        task_cfg,
        device,
        task_id,
        task_count,
        task_iter_train,
        task_dataloader_train,
        model,
        task_losses,
        threshold,
        entropy_threshold=1.5,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    # get the batch
    batch = task_iter_train[task_id].next()
    batch = tuple(t.cuda(device=device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
    # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    # weights = load_label_weights("/root/vilbert-multi-task/datasets/VQA/cache/label_frequencies.pkl")
    # weights = weights.to(device)
    if task_id == "TASK4" or task_id == "TASK17":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, image_id, is_ambiguous = (
            batch
        )

    batch_size = features.size(0)
    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )
        co_attention_mask = co_attention_mask.view(
            rbatch_size,
            co_attention_mask.size(2),
            co_attention_mask.size(3),
            co_attention_mask.size(4),
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    text_emb, image_emb, vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, ambiguous_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
        task_tokens,
    )

    # for different task, we use different output to calculate the loss.
    if task_cfg[task_id]["type"] == "VL-classifier":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        cls_loss = loss_fn(vil_prediction, target).mean()
        true_labels_count = target.sum(dim=1)
        preds = torch.sigmoid(vil_prediction)
        preds_bin = (preds >= threshold).float()
        sample_scores = torch.zeros(preds.size(0)).to(preds.device)
        results = []
        entropy_list = []
        entropy_acc = []

        image_id_to_objects = torch.load("/root/simmc_2/simmc2/resnet_features_objectlevel.pt")
        ambiguous_labels = batch[10].float().to(preds.device)
        for i in range(preds.size(0)):  # 遍历每个样本
            # sigmoid_vil = torch.sigmoid(vil_prediction)
            pred_i = preds[i]  # 当前样本的预测标签
            current_true_label_count = int(true_labels_count[i].item())  # 当前样本真实标签的数量
            if pred_i.sum() < current_true_label_count:  # 如果预测标签数量少于真实标签数量
                # 获取当前样本的前 true_labels_count 个最大概率的标签
                top_k_indices = torch.topk(pred_i, current_true_label_count).indices
                # preds[i, top_k_indices] = 1.0
            # sample_scores[i] = (preds[i] == target[i]).all().float()
            # score == correct.sum() / correct.size(0)
            # return scores
            intersection = (pred_i * target[i]).sum().float()
            union = (pred_i + target[i]).clamp(0, 1).sum().float()  # 并集
            sample_scores[i] = intersection / (union + 1e-6)
            filtered_sigmoid = pred_i[pred_i >= threshold]
            if filtered_sigmoid.numel() == 0:
                k = max(1, int(current_true_label_count))
                topk_vals = torch.topk(pred_i, k=min(k, pred_i.size(0))).values
                norm_probs = topk_vals / (topk_vals.sum() + 1e-12)
            else:
                norm_probs = filtered_sigmoid / (filtered_sigmoid.sum() + 1e-12)

            entropy_i = -(norm_probs * (norm_probs + 1e-12).log()).sum()
            entropy_list.append(entropy_i)
        entropy_tensor = torch.stack(entropy_list)  # [batch_size]
        num_answers = target[0].sum().item()

        text_emb = text_emb
        image_emb = image_emb
        txt_feat = F.normalize(text_emb, dim=-1)
        img_feat = F.normalize(image_emb, dim=-1)
        text_image_similarity = (txt_feat * img_feat).sum(dim=-1)
        total_loss = cls_loss
        correct_preds = 0
        for i in range(preds.size(0)):
            question_str = question_id[i]
            entropy_val = entropy_tensor[i]
            num_ans = target[i].sum().item()
            pred_answer_count = (pred_i >= threshold).sum().item()
            entropy_alignment_loss_i, spatial_amb, score_diff, exp_entropy, ambiguous_flag_i, language_guided = is_ambiguous_full(
                question=question_str,
                object_scores=preds[i],
                bboxes=spatials[i][:, :4],
                entropy=entropy_val,
                num_answers=num_ans,
                pred_answer_count=pred_answer_count,
                entropy_threshold=entropy_threshold,
            )
            lambda_entropy = 1.0 if mentions_multiple_objects(question_str) or num_ans >= 2 else 0.2
            loss_i = lambda_entropy * entropy_alignment_loss_i
            total_loss = total_loss + loss_i
            #             ambiguous_logit_i = ambiguous_logit[i] if ambiguous_logit is not None else None

            #             loss_i = compute_loss_with_entropy(
            #                 cls_loss, entropy_alignment_loss_i, ambiguous_flag_i, is_ambiguous[i].item(), ambiguous_logit_i, lambda_entropy
            #             )

            #             entropy_acc.append(float(ambiguous_flag_i == bool(is_ambiguous[i].item())))
            rule_feats = torch.tensor([
                float(entropy_alignment_loss_i > 1.0),
                float(spatial_amb),
                float(not language_guided and pred_answer_count >= 2),
                float(not language_guided and exp_entropy > 1.7),
                float(score_diff < 0.1)
            ], device=vil_prediction.device).unsqueeze(0)
            ambiguity_logit = model.module.ambiguity_classifier(rule_feats)
            pred_amb = (torch.sigmoid(ambiguity_logit) >= 0.5).float().item()
            label = is_ambiguous[i].item()
            correct_preds = correct_preds + (pred_amb == label)
            ambiguity_gt = torch.tensor([float(is_ambiguous[i].item())],
                                        device=rule_feats.device)  # True or False label
            ambiguity_loss = F.binary_cross_entropy_with_logits(ambiguity_logit.view(-1), ambiguity_gt)
            total_loss = total_loss + 1.2 * ambiguity_loss
            result = {
                "entropy": entropy_val.item(),
                "ambiguous": ambiguous_flag_i,
                "lambda_entropy": lambda_entropy,
                "num_answers": num_ans,
                "question_id": question_str,
                "image_id": image_id[i],
                "cls_loss": cls_loss.item(),
                "entropy_alignment_loss": entropy_alignment_loss_i,
                "text_image_similarity": text_image_similarity[i].item(),
                "predicted_ambiguous": ambiguous_flag_i,
                "groundtruth_ambiguous": is_ambiguous[i].item(),
                "spatial_amb": spatial_amb,
                "score_diff": score_diff,
                "exp_entropy": exp_entropy,

                "language_guided": language_guided,
                "ambiguity_logit": ambiguity_logit,
                "ambiguity_loss": ambiguity_loss,
                "ambiguous_by_rule": ambiguous_flag_i,
                "ambiguous_by_model": (torch.sigmoid(ambiguity_logit).item() > 0.5),
            }

            result = update_metrics(result, ambiguous_flag_i, is_ambiguous[i].item())
            results.append(result)
        total_loss = total_loss / preds.size(0)
        float_score = correct_preds / batch_size
    #         float_score = sum(entropy_acc) / len(entropy_acc)

    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        loss = task_losses[task_id](vil_prediction_gqa, target)
        loss = loss.mean() * target.size(1)
        batch_score = compute_score_with_logits(
            vil_prediction_gqa, target
        ).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = float(torch.sum(select_target > 0.5)) / batch_size

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum()) / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(
            vil_binary_prediction, target
        ).sum() / float(batch_size)

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(
            vil_tri_prediction, target
        ).sum() / float(batch_size)

    return total_loss, float_score, results


def LoadLosses(args, task_cfg, task_ids):
    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="trainval"):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    features_h5path1 = "/root/simmc1/datase/gt0313"
    features_h5path2 = "/root/simmc1/datase/gt0313"
    task_feature_reader1[features_h5path1] = ImageFeaturesH5Reader(features_h5path1, args.in_memory)
    task_feature_reader2[features_h5path2] = ImageFeaturesH5Reader(features_h5path2, args.in_memory)
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_cfg[task]["name"]
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        # num_workers = int(num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
                split=task_cfg[task]["train_split"],
                image_features_reader=task_feature_reader1[features_h5path1],
                gt_image_features_reader=task_feature_reader2[features_h5path2],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                # clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=23,
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = DatasetMapTrain[task_name](
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader=task_feature_reader1[features_h5path1],
                gt_image_features_reader=task_feature_reader2[features_h5path2],
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                # clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=23,
                max_region_num=task_cfg[task]["max_region_num"],
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                # TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task])

            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True,
            )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, ids):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    features_h5path1 = "/root/simmc1/datase/gt0313"
    features_h5path2 = "/root/simmc1/datase/gt0313"

    task_feature_reader1[features_h5path1] = ImageFeaturesH5Reader(features_h5path1, args.in_memory)
    task_feature_reader2[features_h5path2] = ImageFeaturesH5Reader(features_h5path2, args.in_memory)

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_ids.append(task)
        task_name = task_cfg[task]["name"]
        batch_size = args.batch_size
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())

        num_workers = int(args.num_workers / len(ids))
        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        if args.split:
            eval_split = args.split
        else:
            eval_split = task_cfg[task]["val_split"]

        task_datasets_val[task] = DatasetMapEval[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=eval_split,
            image_features_reader=task_feature_reader1[features_h5path1],
            gt_image_features_reader=task_feature_reader2[features_h5path2],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            # clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=23,
            max_region_num=task_cfg[task]["max_region_num"],
        )

        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=10,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_val[task])
        task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )
def EvaluatingModel(
        args,
        task_cfg,
        device,
        task_id,
        batch,
        model,
        task_dataloader,
        task_losses,
        results,
        others,
        threshold
):
    # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    batch = tuple(t.cuda(device=device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
    if task_id == "TASK4" or task_id == "TASK17":
        features, spatials, image_mask, question, target, input_mask, segment_ids, multiple_choice_ids, co_attention_mask, question_id = (
            batch
        )
    else:
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, is_ambiguous = (
            batch
        )
    batch_size = features.size(0)

    if task_cfg[task_id]["process"] in ["dialog"]:
        max_num_bbox = features.size(1)
        nround = question.size(1)
        num_options = question.size(2)
        rbatch_size = batch_size * nround
        question = question.view(rbatch_size, question.size(2), question.size(3))
        target = target.view(-1)
        input_mask = input_mask.view(
            rbatch_size, input_mask.size(2), input_mask.size(3)
        )
        segment_ids = segment_ids.view(
            rbatch_size, segment_ids.size(2), segment_ids.size(3)
        )
        co_attention_mask = co_attention_mask.view(
            rbatch_size,
            co_attention_mask.size(2),
            co_attention_mask.size(3),
            co_attention_mask.size(4),
        )

        features = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, nround, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )

        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )
        batch_size = rbatch_size

    elif task_cfg[task_id]["process"] in ["expand"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = (
            features.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 2048)
            .contiguous()
            .view(-1, max_num_bbox, 2048)
        )
        spatials = (
            spatials.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox, 5)
            .contiguous()
            .view(-1, max_num_bbox, 5)
        )
        image_mask = (
            image_mask.unsqueeze(1)
            .expand(batch_size, num_options, max_num_bbox)
            .contiguous()
            .view(-1, max_num_bbox)
        )
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["retrieval"]:
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(-1, features.size(2), features.size(3))
        spatials = spatials.view(-1, spatials.size(2), spatials.size(3))
        image_mask = image_mask.view(-1, image_mask.size(2))
        question = question.view(-1, question.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(
            -1, co_attention_mask.size(2), co_attention_mask.size(3)
        )

    elif task_cfg[task_id]["process"] in ["nlvr"]:
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = question.size(1)
        features = features.view(
            batch_size * 2, int(features.size(1) / 2), features.size(2)
        )
        spatials = spatials.view(
            batch_size * 2, int(spatials.size(1) / 2), spatials.size(2)
        )
        image_mask = image_mask.view(batch_size * 2, int(image_mask.size(1) / 2))
        question = question.repeat(1, 2)
        question = question.view(batch_size * 2, int(question.size(1) / 2))
        input_mask = input_mask.repeat(1, 2)
        input_mask = input_mask.view(batch_size * 2, int(input_mask.size(1) / 2))
        segment_ids = segment_ids.repeat(1, 2)
        segment_ids = segment_ids.view(batch_size * 2, int(segment_ids.size(1) / 2))
        co_attention_mask = co_attention_mask.view(
            batch_size * 2,
            int(co_attention_mask.size(1) / 2),
            co_attention_mask.size(2),
        )

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))

    with torch.no_grad():
        text_emb, image_emb, vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
            task_tokens,
        )

    if task_cfg[task_id]["type"] == "VL-classifier":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        cls_loss = loss_fn(vil_prediction, target)
        cls_loss = cls_loss.mean() * target.size(1)
        preds = torch.sigmoid(vil_prediction)
        # threshold = 0.5
        preds = (preds > threshold).float()
        # target = (target > threshold).float()
        k = 3
        min_threshold = 0.3
        entropy_list = []
        results = []
        for i in range(preds.size(0)):  # 遍历每个样本
            predicted_labels = preds[i]  # 当前样本的预测标签
            # current_true_label_count = int(true_labels_count[i].item())  # 当前样本真实标签的数量
            if predicted_labels.sum() == 0:  # 如果预测标签数量少于真实标签数量
                # 获取当前样本的前 true_labels_count 个最大概率的标签
                adjusted_preds = (torch.sigmoid(vil_prediction[i]) > min_threshold).float()
                # 降低阈值后可能输出多个标签，取 top-k 最大概率的标签
                if adjusted_preds.sum() > 0:  # 如果低阈值后有标签
                    dynamic_k = min(k, int(adjusted_preds.sum().item()))
                    top_k_indices = torch.topk(adjusted_preds, dynamic_k).indices
                    adjusted_preds = torch.zeros_like(adjusted_preds)  # 初始化为全 0
                    adjusted_preds[top_k_indices] = 1.0  # 只保留 top-k 标签

                preds[i] = adjusted_preds
            intersection = (preds[i] * target[i]).sum().float()  # 交集 (True Positives)
            union = (preds[i] + target[i]).clamp(0, 1).sum().float()  # 并集
            sample_scores[i] = intersection / (union + 1e-6)
            pred_i = preds[i]
            entropy_i = -(pred_i * (pred_i + 1e-12).log()).sum()
            entropy_list.append(entropy_i)
        batch_score = sample_scores.mean()

        results.append(
            {
                "question": question_id[i].item(),
                "answer": answers,
                "preds": tensor_to_list(preds[i]),
                "batch_score": tensor_to_list(batch_score),

            }
        )
        txt_feat = pooled_output_t  # [batch, hidden]
        img_feat = pooled_output_v  # [batch, hidden]
        txt_feat = F.normalize(txt_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)
        similarity = (txt_feat * img_feat).sum(dim=-1)
        ambiguous_labels = batch[9].float().to(preds.device)
        entropy_tensor = torch.stack(entropy_list).detach()  # [batch_size]
        sim_tensor = similarity.detach()
        entropy_logit = entropy_tensor.unsqueeze(1)  # [B, 1]
        #         if not hasattr(self, "entropy_classifier"):
        #              self.entropy_classifier = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, 1))
        #         entropy_pred = self.entropy_classifier(entropy_logit).squeeze(1)
        entropy_pred = model.module.entropy_classifier(entropy_logit).squeeze(1)
        entropy_loss = torch.nn.BCEWithLogitsLoss()(entropy_pred, ambiguous_labels)
        entropy_acc = ((torch.sigmoid(entropy_pred) >= 0.5).float() == ambiguous_labels).float().mean()

        # （2）仅相似度预测模糊
        sim_logit = sim_tensor.unsqueeze(1)  # [B, 1]
        #         if not hasattr(self, "sim_classifier"):
        #             self.sim_classifier = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, 1))
        #         sim_pred = self.sim_classifier(sim_logit).squeeze(1)
        sim_pred = model.module.sim_classifier(sim_logit).squeeze(1)
        sim_loss = torch.nn.BCEWithLogitsLoss()(sim_pred, ambiguous_labels)
        sim_acc = ((torch.sigmoid(sim_pred) >= 0.5).float() == ambiguous_labels).float().mean()

        # （3）联合预测（只要一个为真就预测为模糊）
        union_pred = torch.max(torch.sigmoid(entropy_pred), torch.sigmoid(sim_pred))
        union_pred_logits = torch.log(union_pred / (1 - union_pred + 1e-12) + 1e-12)
        union_loss = torch.nn.BCEWithLogitsLoss()(union_pred_logits, ambiguous_labels)
        union_acc = ((union_pred >= 0.5).float() == ambiguous_labels).float().mean()

        # 模糊预测准确率（联合）
        pred_ambiguous = (union_pred >= 0.5).float()
        ambiguity_accuracy = (pred_ambiguous == ambiguous_labels).float().mean()
        loss_dict = {
            "cls_loss": cls_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "entropy_acc": entropy_acc.item(),
            "sim_loss": sim_loss.item(),
            "union_loss": union_loss.item(),
            "union_acc": union_acc.item(),
            "ambiguity_acc": ambiguity_accuracy.item(),
            "mean_entropy": entropy_tensor.mean().item(),
            "mean_sim": sim_tensor.mean().item(),
            "batch_score(IOU)": batch_score.item()
        }
        loss = cls_loss
    #         for i in range(preds.size(0)):
    #             results.append({
    #                 "question_id": question_id[i],
    #                 "is_ambiguous_gt": ambiguous_labels[i].item(),
    #                 "is_ambiguous_pred": pred_ambiguous[i].item(),

    #                 "entropy": entropy_tensor[i].item(),
    #                 "similarity": sim_tensor[i].item()
    #               })
    elif task_cfg[task_id]["type"] == "VL-classifier-GQA":
        logits = torch.max(vil_prediction_gqa, 1)[1].data
        loss = 0
        batch_score = 0
        for i in range(logits.size(0)):
            results.append(
                {
                    "questionId": str(question_id[i].item()),
                    "prediction": task_dataloader[task_id].dataset.label2ans[
                        logits[i].item()
                    ],
                }
            )

    elif task_cfg[task_id]["type"] == "VL-logit":
        vil_logit = vil_logit.view(batch_size, num_options)
        loss = task_losses[task_id](vil_logit, target)
        _, preds = torch.max(vil_logit, 1)
        batch_score = (preds == target).sum()

        probs = torch.softmax(vil_logit, dim=1)
        for i in range(vil_logit.size(0)):
            results.append(
                {
                    "question_id": question_id[i].item(),
                    "answer": [prob.item() for prob in probs[i]],
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit":
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, select_idx = torch.max(vision_logit, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        batch_score = torch.sum(select_target > 0.5).item()

        for i in range(select_idx.size(0)):
            results.append(
                {
                    "id": question_id[i].item(),
                    "target": select_idx[i].item(),
                    "IOU": select_target[i].item(),
                }
            )

    elif task_cfg[task_id]["type"] == "V-logit-mc":
        vision_logit = vision_logit[:, 101:]
        vision_logit = vision_logit.squeeze(2).gather(1, multiple_choice_ids)
        vision_logit = vision_logit.unsqueeze(2)
        loss = task_losses[task_id](vision_logit, target)
        loss = loss.mean() * target.size(1)
        _, preds = torch.max(vision_logit, dim=1)
        _, target = torch.max(target, dim=1)
        batch_score = float((preds == target).sum())

        for i in range(preds.size(0)):
            results.append({"id": question_id[i].item(), "target": preds[i].item()})

    elif task_cfg[task_id]["type"] == "VL-binary-classifier":
        loss = task_losses[task_id](vil_binary_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_binary_prediction, target).sum()

    elif task_cfg[task_id]["type"] == "VL-tri-classifier":
        loss = task_losses[task_id](vil_tri_prediction, target)
        loss = loss.mean()
        batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return loss_dict, batch_size, results, others, loss
