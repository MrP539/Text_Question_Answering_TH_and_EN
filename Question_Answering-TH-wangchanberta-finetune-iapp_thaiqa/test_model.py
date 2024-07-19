from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "MRP101py/QA-wangchanberta-finetune-iapp_thaiqa"
question_answerer = pipeline("question-answering", model=model_checkpoint)
context = """
การท่องเที่ยวเป็นกิจกรรมที่ได้รับความนิยมอย่างสูงเนื่องจากช่วยให้ผู้คนได้สัมผัสกับวัฒนธรรมใหม่ ๆ และขยายขอบเขตความรู้เกี่ยวกับสถานที่ต่าง ๆ การเดินทางไปยังสถานที่ที่ไม่เคยไปมาก่อนสามารถช่วยเพิ่มประสบการณ์ชีวิตและทำให้ผู้คนรู้สึกสดชื่นและมีแรงบันดาลใจใหม่ ๆ การท่องเที่ยวยังสามารถช่วยให้เกิดความสัมพันธ์ที่ดีขึ้นระหว่างบุคคลและสร้างความทรงจำที่มีค่า
"""


question = "การท่องเที่ยวมีประโยชน์อย่างไรต่อประสบการณ์ชีวิตและความสัมพันธ์ระหว่างบุคคล?"

print(question_answerer(question=question, context=context))