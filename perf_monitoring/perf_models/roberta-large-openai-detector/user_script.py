# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os

import torch
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from olive.constants import Framework
from olive.evaluator.accuracy import AccuracyScore
from olive.model import OliveModel

# https://huggingface.co/roberta-large-openai-detector
model_name = "roberta-large-openai-detector"
dataset_name = "glue"
subset = "mnli_matched"
split = "validation"


class CalibrationDataLoader(CalibrationDataReader):
    def __init__(self, dataloader, post_func, num_samplers=100):
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.post_func = post_func
        self.counter = 0
        self.num_samplers = num_samplers

    def get_next(self):
        if self.counter >= self.num_samplers:
            return None
        self.counter += 1
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            return self.post_func(next(self.iter))
        except StopIteration:
            return None

    def rewind(self):
        self.iter = None
        self.counter = 0


# -------------------- model -------------------
def load_model(model_path=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model


# -------------------- dataset -------------------


def create_evaluation_dataset(dataset_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rls_ordered = []
    for item, label in [("small-117M.valid.jsonl", 0), ("webtext.valid.jsonl", 1)]:
        valid_file = os.path.join(dataset_dir, item)
        with open(valid_file, "r") as f:
            for line in f:
                line = json.loads(line)
                input = tokenizer(line["text"], return_tensors="pt", padding=True, truncation=True)
                rls_ordered.append((input, label))

    rls = []
    for i in range(len(rls_ordered) // 2):
        rls.append(
            {
                "input_ids": rls_ordered[i][0].input_ids[0],
                "attention_mask": rls_ordered[i][0].attention_mask[0],
                "labels": rls_ordered[i][1],
            }
        )
        next_i = i + len(rls_ordered) // 2
        rls.append(
            {
                "input_ids": rls_ordered[next_i][0].input_ids[0],
                "attention_mask": rls_ordered[next_i][0].attention_mask[0],
                "labels": rls_ordered[next_i][1],
            }
        )

    class _Dateset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return self.dataset[index], self.dataset[index]["labels"]

        def __len__(self):
            return 5
            # return len(self.dataset)

    return _Dateset(rls)


def create_dataloader(data_dir="data", batch_size=2):
    def _collate_fn(batch):
        batch = default_collate(batch)
        return batch

    dataset = create_evaluation_dataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn)


def create_cali_dataloader():
    def _post_func(sampler):
        return sampler

    dataloader = create_dataloader()
    cali_dataloader = CalibrationDataLoader(create_dataloader(dataloader, _post_func))
    return cali_dataloader


def post_process(output):
    import torch
    import transformers

    if isinstance(output, transformers.modeling_outputs.SequenceClassifierOutput):
        pre = torch.argmax(output.logits, dim=-1)
    else:
        pre = torch.argmax(output, dim=-1)
    return pre


def eval_accuracy(model: OliveModel, data_dir, batch_size, device, execution_providers):
    dataloader = create_dataloader(data_dir, batch_size)
    preds = []
    target = []
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)
    if model.framework == Framework.ONNX:
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        for inputs, labels in dataloader:
            if isinstance(inputs, dict):
                input_dict = {k: inputs[k].tolist() for k in inputs.keys()}
            else:
                inputs = inputs.tolist()
                input_dict = dict(zip(input_names, [inputs]))
            res = sess.run(input_feed=input_dict, output_names=None)
            if len(output_names) == 1:
                result = torch.Tensor(res[0])
            else:
                result = torch.Tensor(res)
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())
    elif model.framework == Framework.PYTORCH:
        for inputs, labels in dataloader:
            if isinstance(inputs, dict):
                result = sess(**inputs)
            else:
                result = sess(inputs)
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())
    return AccuracyScore().measure(preds, target)
