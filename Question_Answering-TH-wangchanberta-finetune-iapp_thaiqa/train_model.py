import transformers
import datasets
from tqdm.auto import tqdm
import collections
import numpy as np
import os
import pandas as pd

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

## ฉhoice Pretrained Model  ##

# model_names = [
#     'wangchanberta-base-att-spm-uncased',
#     'xlm-roberta-base',
#     'bert-base-multilingual-cased',
#     'wangchanberta-base-wiki-newmm',
#     'wangchanberta-base-wiki-ssg',
#     'wangchanberta-base-wiki-sefr',
#     'wangchanberta-base-wiki-spm',
# ]

# tokenizers = {
#     'wangchanberta-base-att-spm-uncased': AutoTokenizer,
#     'xlm-roberta-base': AutoTokenizer,
#     'bert-base-multilingual-cased': AutoTokenizer,
#     'wangchanberta-base-wiki-newmm': ThaiWordsNewmmTokenizer,
#     'wangchanberta-base-wiki-ssg': ThaiWordsSyllableTokenizer,
#     'wangchanberta-base-wiki-sefr': FakeSefrCutTokenizer,
#     'wangchanberta-base-wiki-spm': ThaiRobertaTokenizer,
    
# }                     


public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased'] 


#parameterizing columns to complie code
CONTEXT_COL = 'context'
QUESTION_COL = 'question'
ANSWERS_COL = 'answers'
TEXT_COL = 'text'
START_COL = 'answer_start'
END_COL = 'answer_end'
QUESTION_ID_COL = 'question_id'

########################################################################################  load & setup Datasets  ##########################################################################################

path_datasets = "iapp_thaiqa\iapp_thaiqa"

raw_datasets = datasets.load_from_disk(os.path.join(path_datasets))
        # ชุดข้อมูล "iapp_thaiqa" นั้นเป็นชุดข้อมูลที่ใช้สำหรับงานตอบคำถามภาษาไทย (Thai Question Answering) 
model_checkpoint = "wangchanberta-base-att-spm-uncased"
public_models = ['xlm-roberta-base', 'bert-base-multilingual-cased']

#lowercase when using uncased model
def lowercase_example(example):
    example[QUESTION_COL] =  example[QUESTION_COL].lower()
    example[CONTEXT_COL] =  example[CONTEXT_COL].lower()
    example[ANSWERS_COL][TEXT_COL] =  [example[ANSWERS_COL][TEXT_COL][0].lower()]
    return example

if model_checkpoint == 'wangchanberta-base-att-spm-uncased':
    raw_datasets =raw_datasets.map(lowercase_example)


#tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_checkpoint_or_path=model_checkpoint)


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

#def compute_metrics(start_logits, end_logits, features, examples):

def compute_metrics(start_logits, end_logits, features, examples):
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
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)




################################################################################################  Create CSVlogger  ####################################################################################################

output_dir = os.path.join("result_logging")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class CSVlogger(transformers.TrainerCallback):

    def __init__(self,output_dir):
        self.output_dir = output_dir
        self.history_log = []

    def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl,logs=None, **kwargs):

        if logs == None:
            logs={}

        self.history_log.append(logs)
        self.save_log()

    def save_log(self):
        df = pd.DataFrame(self.history_log)
        df.to_csv((os.path.join(self.output_dir,"result_logging.csv")),index=False)

csv_logger = CSVlogger(output_dir=output_dir)




###################################################################################################  Create model  ############################################################################################################

#model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_checkpoint,revision="main")
model = transformers.AutoModelForQuestionAnswering.from_pretrained(f'airesearch/{model_checkpoint}' if model_checkpoint not in public_models else f'{model_checkpoint}',revision='main')
    # การกำหนด revision='main' เป็นการระบุเวอร์ชันหรือการอ้างอิงที่ต้องการใช้งานจาก Git repository ที่เก็บโมเดลหรือโค้ดต่าง ๆ โดยทั่วไป main หมายถึง branch หลักใน Git

    # ในตอนที่โมเดลได้รับ input_ids จะมีการทำ Embedding ภายในโมเดลโดยอัตโนมัติเมื่อเราทำการฝึกหรือประเมินผล:
###################################################################################################  finetune model  ############################################################################################################


batch_size = 8
#logging_step = len(raw_datasets["train"])//batch_size
logging_step= 500


training_arg = transformers.TrainingArguments(
    output_dir="QA-wangchanberta-finetune-iapp_thaiqa",
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    push_to_hub = True,
    logging_steps=logging_step,
    #load_best_model_at_end=True
)

trainer = transformers.Trainer(
    model=model,
    args=training_arg,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    callbacks=[csv_logger],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\n..... training ......\n")

trainer.train()

print("\n..... training Complete ......\n")

