import torch.utils
import torch.utils.data
import transformers
import datasets
from tqdm.auto import tqdm
import collections
import numpy as np
import os
import pandas as pd
import torch

#thai2transformers
import thai2transformers
from thai2transformers.metrics import (
    squad_newmm_metric,
    question_answering_metrics,
    
)
from thai2transformers.preprocess import (
    prepare_qa_train_features
)
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer,
    SEFR_SPLIT_TOKEN
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#parameterizing columns to complie code
CONTEXT_COL = 'context'
QUESTION_COL = 'question'
ANSWERS_COL = 'answers'
TEXT_COL = 'text'
START_COL = 'answer_start'
END_COL = 'answer_end'
QUESTION_ID_COL = 'question_id'

data_path = os.path.join("iapp_thaiqa\iapp_thaiqa")
raw_datasets = datasets.load_from_disk(data_path)

model_checkpoint = 'wangchanberta-base-att-spm-uncased'



public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased'] 

def lowercase_example(example):
    example[QUESTION_COL] =  example[QUESTION_COL].lower()
    example[CONTEXT_COL] =  example[CONTEXT_COL].lower()
    example[ANSWERS_COL][TEXT_COL] =  [example[ANSWERS_COL][TEXT_COL][0].lower()]
    return example

if model_checkpoint == 'wangchanberta-base-att-spm-uncased':
    raw_datasets =raw_datasets.map(lowercase_example)

tokenizer = transformers.AutoTokenizer.from_pretrained(
                f'airesearch/{model_checkpoint}' if model_checkpoint not in public_models else f'{model_checkpoint}',
                revision='main',
                model_max_length=416,)

max_length = 384
stride = 128

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["question_id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

train_dataset = raw_datasets["train"].map(preprocess_training_examples,batched=True,remove_columns=raw_datasets["train"].column_names)
validation_dataset = raw_datasets["validation"].map(preprocess_validation_examples,batched=True,remove_columns=raw_datasets["validation"].column_names)


################################################################################################  Create metrics  ##################################################################################################

metric = squad_newmm_metric # โหลด F1 กับ exact match มา

n_best = 20
max_answer_length = 30

def compute_metrics(start_logits, end_logits, features, examples):
    print("\n... Straat Add Batch to Metric ...\n")
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["question_id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["question_id"], "answers": {'text': ex[ANSWERS_COL][TEXT_COL],
                           'answer_start':ex[ANSWERS_COL][START_COL]}} for ex in examples]
    metric.add_batch(predictions=predicted_answers, references=theoretical_answers)
    return(print("\n... Add Batch to Metric Complete ...\n"))


#########################################################################     Evaluate   ALL Data on ones      #####################################################################################

# small_eval_set = raw_datasets["validation"].select(range(200)) # ดาต้าขนาดเล็กก็พอแล้ว
# #trained_checkpoint = "airesearch/wangchanberta-base-wiki-20210520-spm-finetune-qa" # โมเดลที่เทรนมาเรียบร้อยแล้ว

# eval_set = small_eval_set.map(
#     preprocess_validation_examples,
#     batched=True,
#     remove_columns=raw_datasets["validation"].column_names,
# )

# eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])

# eval_set_for_model.set_format("torch")


# #dl = torch.utils.data.DataLoader(eval_set_for_model,batch_size=True,num_workers=0,shuffle=False)


# batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}

# model = transformers.AutoModelForQuestionAnswering.from_pretrained(f'airesearch/{model_checkpoint}' if model_checkpoint not in public_models else f'{model_checkpoint}',revision='main')
#     # การกำหนด revision='main' เป็นการระบุเวอร์ชันหรือการอ้างอิงที่ต้องการใช้งานจาก Git repository ที่เก็บโมเดลหรือโค้ดต่าง ๆ โดยทั่วไป main หมายถึง branch หลักใน Git
# model.to(device)

# with torch.no_grad():
#     outputs = model(**batch) # ได้คู่ของ start,end ที่มาจากโมเดลที่เทรนมาเรียบร้อยแล้ว!

# start_logits = outputs.start_logits.cpu().numpy()
# end_logits = outputs.end_logits.cpu().numpy() # เอาออกจาก gpu เพราะว่าแค่เอาเลขมาเทียบ ไม่จำเป็นต้องลง gpu

# compute_metrics(start_logits=start_logits,end_logits=end_logits,features=eval_set,examples=small_eval_set)
# print(metric.compute())

########################################################################     Evaluate  whit batch      #####################################################################################

model = transformers.AutoModelForQuestionAnswering.from_pretrained(f'airesearch/{model_checkpoint}' if model_checkpoint not in public_models else f'{model_checkpoint}',revision='main')


# small_eval_set = raw_datasets["validation"].select(range(200)) # ดาต้าขนาดเล็กก็พอแล้ว
# #trained_checkpoint = "airesearch/wangchanberta-base-wiki-20210520-spm-finetune-qa" # โมเดลที่เทรนมาเรียบร้อยแล้ว


# eval_set = small_eval_set.map(
#     preprocess_validation_examples,
#     batched=True,
#     remove_columns=raw_datasets["validation"].column_names,
# )

# print(small_eval_set[0:3])
# print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")
# print(eval_set[4:8])

eval_set_for_model = validation_dataset.remove_columns(["example_id","offset_mapping"])
eval_set_for_model.set_format("torch")

batch_size = 8
eval_data_loader  =  torch.utils.data.DataLoader(eval_set_for_model,batch_size=batch_size,num_workers=0,shuffle=False)

all_start_logits = []
all_end_logits = []

model.to(device)
for batchs in tqdm(eval_data_loader):
    
    batch = {k:v.to(device) for k,v in batchs.items() }
    
    with torch.no_grad():
        outputs=model(**batch)
    
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy() # เอาออกจาก gpu เพราะว่าแค่เอาเลขมาเทียบ ไม่จำเป็นต้องลง gpu
    all_start_logits.append(start_logits)
    all_end_logits.append(end_logits)
    

# รวบรวมผลลัพธ์ทั้งหมด
all_start_logits = np.concatenate(all_start_logits, axis=0)
all_end_logits = np.concatenate(all_end_logits, axis=0)

    
    
compute_metrics(start_logits=all_start_logits,end_logits=all_end_logits,features=validation_dataset,examples=raw_datasets["validation"])
print(f'\nBefor Training {metric.compute()}\n')

########################################################################################################################################################################################


# Befor Training {'exact_match': 0.0, 'f1': 8.638442671340206}
