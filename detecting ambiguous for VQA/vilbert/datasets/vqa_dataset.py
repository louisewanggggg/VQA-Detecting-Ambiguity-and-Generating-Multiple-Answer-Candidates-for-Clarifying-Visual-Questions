import torch
import numpy as np
import os
import json
import pickle as cPickle
from torch.utils.data import Dataset
from ._image_features_reader import ImageFeaturesH5Reader


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, image_id, answer, question_id, is_ambiguous, object_map):
    question = question.strip()
    if isinstance(answer, list):
        answer = {
            "labels": answer,
            "scores": [1.0] * len(answer)
        }
    entry = {
        "image_id": image_id,
        "question": question,
        "answer": answer,
        "question_id": question_id,
        "is_ambiguous": is_ambiguous,
        "object_map": object_map
    }
    return entry


def _load_dataset(dataroot, name):
    scene_json_folder = "/root/simmc_2/simmc2/data/public"
    feature_path = "/root/simmc_2/simmc2/resnet_features_objectlevel.pt"
    image_features = torch.load(feature_path)
    if name == "trainval":
        #         data = cPickle.load(open("/root/simmc_data/simmc_train.pkl", 'rb'))
        #         data1 = cPickle.load(open("/root/simmc_data/simmc_trainval.pkl", 'rb'))
        data = cPickle.load(open("/root/simmc2/model/disambiguate/merged_data.pkl", "rb"))
        data1 = cPickle.load(open("/root/simmc2/model/disambiguate/merged_data.pkl", "rb"))
        data = data + data1[:-3000]
    elif name == "train" or name == "val":
        data = cPickle.load(open("/root/simmc2/model/disambiguate/merged_data.pkl", "rb"))
    elif name == "minval":
        data = cPickle.load(open("/root/simmc2/model/disambiguate/merged_data.pkl", "rb"))[-3000:]

    entries = []
    for item in data:
        image_id = item["image_id"].replace(".png", "")
        if image_id not in image_features:
            print(f"[SKIP] Image {image_id} not in feature file. Skipping...")
            continue
        scene_path = os.path.join(scene_json_folder, f"{image_id}_scene.json")
        if not os.path.exists(scene_path):
            continue
        with open(scene_path, "r") as f:
            scene = json.load(f)
        object_map = [obj["index"] for obj in scene["scenes"][0]["objects"]]
        entry = _create_entry(item["question"], image_id, item["answer"], item["question_id"], item["is_ambiguous"],
                              object_map)
        entries.append(entry)
    return entries

def oversample_ambiguous(entries, factor=3):
    ambiguous = [e for e in entries if e["is_ambiguous"] == 1]
    nonambiguous = [e for e in entries if e["is_ambiguous"] == 0]
    augmented = nonambiguous + ambiguous * factor
    random.shuffle(augmented)
    return augmented


class ResnetObjectFeatureReader:
    def __init__(self, feature_path):
        self.data = torch.load(feature_path)

    def __getitem__(self, image_id):
        return self.data[image_id]


class VQAClassificationDataset(Dataset):
    def __init__(self, task, dataroot, annotations_jsonpath, split,
                 image_features_reader, gt_image_features_reader, tokenizer,
                 bert_model, padding_index=0, max_seq_length=23,
                 max_region_num=101):
        super().__init__()
        self.split = split
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = ResnetObjectFeatureReader("/root/simmc_2/simmc2/resnet_features_objectlevel.pt")
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        raw_entries = _load_dataset(dataroot, split)
        if split == "train":
            self.entries = oversample_ambiguous(raw_entries, factor=3)  # 模糊样本增强
        else:
            self.entries = raw_entries
        self.tokenize(max_seq_length)
        self.tensorize()

    def tokenize(self, max_length=256):
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids
    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids
            target = torch.zeros(self._max_region_num)
            for aid in entry["answer"]["labels"]:
                if aid < self._max_region_num:
                    target[aid] = 1.0
            entry["target"] = target

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        object_entries = self._image_features_reader[image_id]
        question_id = entry["question_id"]
        features_list = []
        boxes_list = []
        for obj_idx in sorted(object_entries.keys()):
            obj = object_entries[obj_idx]
            features_list.append(obj["feature"].unsqueeze(0))  # shape: [1, 2048]
            boxes_list.append(torch.tensor(obj["box"] + [1.0]))  # [x1, y1, x2, y2, 1.0]
        features = torch.cat(features_list, dim=0)  # [N, 2048]
        boxes = torch.stack(boxes_list, dim=0)  # [N, 5]
        num_boxes = features.size(0)
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))

        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = entry["target"]
        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
            image_id,
            entry["is_ambiguous"],
        )

    def __len__(self):
        return len(self.entries)
