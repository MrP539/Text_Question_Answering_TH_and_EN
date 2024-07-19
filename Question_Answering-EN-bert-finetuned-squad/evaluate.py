import torch
from torch.utils.data import DataLoader
import transformers  # หรือ library อื่นๆที่คุณใช้
import datasets
from tqdm.auto import tqdm
import collections
import numpy as np
# ... (ส่วนอื่นๆของโค้ด เช่น การโหลดโมเดล การเตรียมข้อมูล etc.) ...

# กำหนด device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################################################################  download $ setup dataset  #############################################################################################

raw_dataset = datasets.load_dataset("squad",trust_remote_code=True)
    #SQuAD (Stanford Question Answering Dataset) คือชุดข้อมูลสำหรับการฝึกและทดสอบโมเดลการตอบคำถาม (Question Answering) ซึ่งพัฒนาโดยมหาวิทยาลัยสแตนฟอร์ด 
    # ชุดข้อมูลนี้ถูกออกแบบมาเพื่อการประเมินความสามารถของโมเดลในการอ่านเข้าใจเนื้อหาในบทความและตอบคำถามจากข้อมูลในบทความนั้น ๆ
print(raw_dataset)
print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")

model_checkpoint = "bert-base-cased"

tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_checkpoint)

max_length = 128
stride =  32

def process_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(text =questions, 
                       text_pair=examples["context"], 
                       max_length=max_length, 
                       truncation="only_second",
                            #ค่า truncation="only_second" เป็นการกำหนดให้ทำการตัดข้อความ (truncate) ในส่วนของข้อความที่สอง (text_pair) 
                            # เมื่อความยาวรวมของข้อความทั้งสองเกิน max_length ที่กำหนดไว้ (ในที่นี้คือ 100 tokens) โดยไม่ทำการตัดข้อความในส่วนแรก (text)
                       stride=stride,
                       return_overflowing_tokens=True,
                            # ตรงนี้หลังจากที่เราแบ่งเป็นก้อนๆตาม max_length ที่กำหนด แล้ว ตัวตัดคำเราจะบอกด้วยว่า ก้อนไหนเป็นของประโยคที่เท่าไหร่บ้าง  ex มี 2 context ---> [0,0,0,0,1,1,1,1]
                            # ซึ่งมีประโยชน์มากๆ เพราะเราจะรู้ว่าก้อนไหนมาจากประโยคไหนบ้าง 
                            # สามารถเช็คได้ผ่าน key: input_ids --> ex  [[101, 170 ...],[154,238, ..],[101, 170 ...],[154,238, ..],[254,xxx,...],[200,xxx,...],[254,xxx,...],[200,xxx,...]]
                            # หากนำมาเทียบกับขอมูลที่ทำoverflow_tokens  [    0,             0,             0,             0,            1,            1,            1,            1       ]
                       return_offsets_mapping=True,
                            # offset mapping จะบอกถึงตำแหน่งเริ่มต้นและสิ้นสุดของแต่ละ token ในข้อความต้นฉบับ
                                # ข้อความต้นฉบับคือ "Hello world " --->  tokenizer แปลงข้อความนี้เป็น token สองตัวคือ ["Hello", "world"] ---> offset mapping จะเป็น [(0, 5), (6, 11)]
                                    # "Hello" เริ่มที่ตำแหน่ง 0 และสิ้นสุดที่ตำแหน่ง 5
                                    # "world" เริ่มที่ตำแหน่ง 6 และสิ้นสุดที่ตำแหน่ง 11
                            # ทำการแมพคำตอบไปที่ก้อนนั้นๆได้ //
                            # //สรุปคือเป็นการบอกตำแหน่งคำตอบของก้อนนั้นโดยจะได้คำตอบ เป็น start_point และ  end_point ex [0,0,0,0,1,1,1,] -->([83, 51, 19, 0, 0, 66, 30, 0], [85, 53, 21, 0, 0, 72, 36, 0])
                            # ที่เห็นว่า 1 ประโยคมีหลายคำตอบนั้น เกิดมาจากคำตอบใน train set มีหลายคำตอบเอง หรือ อยู่กับการ stride(การเลือนข้อมูล)ทำให้เกิดข้อมูล ซ้ำในแต่ละก้อน
                        padding="max_length"
                            )
    offset_mapping = inputs.pop("offset_mapping")
        # ทำการ ย้ายค่าใน inputs["offset_mapping"] ไว้ใน offset_mapping  และลบข้อมูล offset_mapping ใน inputs ออก โดยใช้ pop
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions =[]

    for idx,mapping_values in enumerate(offset_mapping):
        sample_idx = sample_map[idx]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(idx)
            # sequence_ids = inputs.sequence_ids(idx) เป็นคำสั่งที่ใช้เพื่อดึงข้อมูลเกี่ยวกับการจับคู่ระหว่าง token กับส่วนของข้อความต้นฉบับ (เช่น ส่วนที่เป็น questions หรือส่วนที่เป็น context) 
            # ในการประมวลผลการตอบคำถาม (Question Answering) หรือการประมวลผลข้อความอื่นๆ
            # output --> [None,0,0,0,0,None,1,1,1,...,1,None]
                # None: หมายถึง token พิเศษ เช่น [CLS], [SEP] ที่ใช้โดยโมเดลในการแยกส่วนของข้อความหรือจุดเริ่มต้นของการประมวลผล
                # 0: หมายถึง token ที่มาจากข้อความแรก (ในกรณีนี้คือคำถาม)
                # 1: หมายถึง token ที่มาจากข้อความที่สอง (ในกรณีนี้คือบริบท)
        
        # หา start index และ  end_index ของ context

        index = 0

        while sequence_ids[index] != 1:
            index += 1
        start_index_context = index

        while sequence_ids[index] == 1:
            index +=1
        end_index_context = index-1

        # หากคำตอบไม่สมบูนณ์ใน context ป้ายกำกับคือ (0, 0)

        if mapping_values[start_index_context][0] > end_char or mapping_values[end_index_context][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            index=start_index_context

            while index <= end_index_context and mapping_values[index][0] <= start_char:
                index += 1
            start_positions.append(index-1)

            index=end_index_context

            while index >= start_index_context and mapping_values[index][1] >= end_char:
                index -= 1
            end_positions.append(index+1)
        
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return (inputs)

def process_validation_examples(examples):
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
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        #เพิ่มค่า offset_mapping หาก เป็น context
        inputs["offset_mapping"][i] = [
            ox if sequence_ids[k] == 1 else None for k, ox in enumerate(offset)
        ]
    
    inputs["example_id"] = example_ids
    return inputs


    
#train_processed_dataset = raw_dataset["train"].select(range(500)).map(process_training_examples,batched=True,remove_columns=raw_dataset["train"].column_names)
#valid_processed_dataset = raw_dataset["validation"].select(range(100)).map(process_validation_examples,batched=True,remove_columns=raw_dataset["validation"].column_names) # ไม่ต้อมีตำแหน่งเริ่ม-จบ เพราะโมเดลจะต้องทำนายเอง

train_processed_dataset = raw_dataset["train"].map(process_training_examples,batched=True,remove_columns=raw_dataset["train"].column_names)
valid_processed_dataset = raw_dataset["validation"].map(process_validation_examples,batched=True,remove_columns=raw_dataset["validation"].column_names) # ไม่ต้อมีตำแหน่งเริ่ม-จบ เพราะโมเดลจะต้องทำนายเอง

#print(train_processed_dataset)
print("\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\t|\n\t\t\t\tV")
#print(valid_processed_dataset)
# สร้าง DataLoader สำหรับ eval_set_for_model (batch_size = 4)

###########################################################################################################  Metrics  #############################################################################################

#metric = datasets.load_metric("squad") # F1 score & exact match

metric = datasets.load_metric("squad",trust_remote_code=True)

def compute_metrics(start_logits, end_logits, features, examples):
    #print(f"Check_arg start_logits : {start_logits}\nCheck_arg end_logits : {end_logits}\nCheck_arg features : {features}\nCheck_arg examples : {examples}")
    n_best = 20
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
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

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    # print(predicted_answers)
    # print(theoretical_answers)
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

###########################################################################################################  Evaluate  #############################################################################################
model = transformers.AutoModelForQuestionAnswering("MRP101py/QA-bert-finetuned-squad")

eval_set_for_model = DataLoader(
    valid_processed_dataset.remove_columns(["example_id", "offset_mapping"]),
    batch_size=4,
    shuffle=False,
)

# ย้ายโมเดลไปยัง device ที่เลือก
model.to(device)
model.eval()
all_start_logits = []  
all_end_logits = []
# วนลูปประมวลผลแต่ละ batch
for batch in tqdm(eval_set_for_model):
    # ย้าย tensors ใน batch ไปยัง device และ stack เป็น tensor เดียว
    batch = {k: torch.stack([t.to(device) for t in v]) for k, v in batch.items()} 

    # ประมวลผล batch (ปิด gradient calculation เพราะเราแค่ประเมินผล)
    with torch.no_grad():
        outputs = model(**batch)

    # ดึง start_logits และ end_logits ออกมา และย้ายกลับไป cpu
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    all_start_logits.extend(start_logits)
    all_end_logits.extend(end_logits)

    # คำนวณและแสดง metrics
print(compute_metrics(all_start_logits, all_end_logits,valid_processed_dataset, raw_dataset["validation"]))


print("\n...RUN Complete...\n")