# pip install --quiet transformers==4.1.1
# pip install --quiet git+https://github.com/PyTorchLightning/pytorch-lightning
# pip install --quiet tokenizers==0.9.4
# pip install --quiet sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
import pandas as pd
import preprocess as pp
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import pytorch_lightning as pl
import torch
import numpy as np
train_data = pd.read_csv("dataset/wic/mark_NIKL_SKT_WiC_TrainA.csv")
train_data = train_data.astype(str)
train_data = pd.DataFrame(train_data)
val_data = pd.read_csv("dataset/wic/mark_NIKL_SKT_WiC_DevA.csv")
val_data = val_data.astype(str)
val_data = pd.DataFrame(val_data)
# print(train_data["ANSWER"][0])
# print(type(train_data["ANSWER"]))
# train_data[["ANSWER","SENTENCE1","SENTENCE2"]]=train_data[["ANSWER","SENTENCE1","SENTENCE2"]].astype(str)
# train_data.to_csv('mark_NIKL_SKT_WiC_TrainA.csv', index=False)
# val_data[["ANSWER","SENTENCE1","SENTENCE2"]]=val_data[["ANSWER","SENTENCE1","SENTENCE2"]].astype(str)
# val_data.to_csv('mark_NIKL_SKT_WiC_DevA.csv', index=False)
# print(train_data["ANSWER"][0])
# print(type(train_data["ANSWER"][0]))
model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# text, answer = [], []
# for _, row in train_data.iterrows():
#     cnt = len(tokenizer.encode(row["SENTENCE1"]))
#     text.append(cnt)
#     cnt = len(tokenizer.encode(row["SENTENCE2"]))
#     text.append(cnt)
#     cnt=len(tokenizer.encode(row["ANSWER"]))
#     answer.append(cnt)
# max_len=max(text)
# tar_max=max(answer)
# print(max_len,tar_max)
#
# max_len=512
# tar_max=30





class BoolqDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 111,
        target_max_token_len: int = 4
                 ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encoding = tokenizer(
            data_row["SENTENCE1"],
            data_row["SENTENCE2"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        target_encoding = tokenizer(
            data_row["ANSWER"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100
        return dict(
            sentence1=data_row["SENTENCE1"],
            sentence2=data_row["SENTENCE2"],
            answer_text=data_row["ANSWER"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten()

        )


class BoolqDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer: T5Tokenizer,
            batch_size: int = 8,
            source_max_token_len: int = 111,
            target_max_token_len: int = 4
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = BoolqDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = BoolqDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )



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



if __name__=="__main__":
    torch.multiprocessing.freeze_support()
    pd.set_option('mode.chained_assignment', None)
    pl.seed_everything(42)

    BATCH_SIZE = 2
    N_EPOCHS = 6
    data_module = BoolqDataModule(train_data, val_data, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = BoolqModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=True
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30

    )

    trainer.fit(model, data_module)

