from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
import pandas as pd
import preprocess as pp
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import pytorch_lightning as pl
import torch
import json
pd.set_option('mode.chained_assignment', None)
pl.seed_everything(42)


model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)



class BoolqModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)



def run():
    torch.multiprocessing.freeze_support()
    trained_model = BoolqModel.load_from_checkpoint("새 폴더/epoch=4-step=9164.ckpt")
    trained_model.freeze()

    def generate_answer(question):
        source_encoding = tokenizer(
            question["Question"],
            question["Text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        generated_ids = trained_model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            max_length=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True
        )
        preds = [
            tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        return "".join(preds)


    test_data = pd.read_csv("dataset/boolq/SKT_BoolQ_Dev.tsv", sep="\t")
    cnt=0
    for i in range(len(test_data)):
        sample_question = test_data.iloc[i]
        if sample_question["Answer(FALSE = 0, TRUE = 1)"]==1:
            cnt+=1
        # sample_question = test_data.iloc[i]
        # gen_answer=generate_answer(sample_question)
        # print(sample_question["Answer(FALSE = 0, TRUE = 1)"],gen_answer)
        # if sample_question["Answer(FALSE = 0, TRUE = 1)"]:
        #     answer="TRUE"
        # else:
        #     answer="FALSE"
        # if answer == gen_answer:
        #     cnt+=1
    print("accuracy: ",cnt/len(test_data))
if __name__=="__main__":
    run()