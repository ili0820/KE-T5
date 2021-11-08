# pip install --quiet transformers==4.1.1
# pip install --quiet git+https://github.com/PyTorchLightning/pytorch-lightning
# pip install --quiet tokenizers==0.9.4
# pip install --quiet sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
import pandas as pd
import preprocess as pp
from torch.utils.data import Dataset,DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint,BaseFinetuning
import torch.optim as optim
import pytorch_lightning as pl
import torch

train_data, val_data,test_data, max_len,tar_max = pp.preprocess()
model_name = 'KETI-AIR/ke-t5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)


text, answer = [], []
for _, row in train_data.iterrows():
    cnt = len(tokenizer.encode(row["Text"]))
    text.append(cnt)

    cnt=len(tokenizer.encode(row["answer_text"]))
    answer.append(cnt)
max_len=max(text)
tar_max=max(answer)+1
print(max_len,tar_max)

class BoolqDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int =max_len,
        target_max_token_len: int =tar_max
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
            data_row["Question"],
            data_row["Text"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        target_encoding = tokenizer(
            data_row["answer_text"],
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
            question=data_row["Question"],
            context=data_row["Text"],
            answer_text=data_row["answer_text"],
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
            batch_size: int = 1,
            source_max_token_len: int =max_len,
            target_max_token_len: int =tar_max
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
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
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
        return AdamW(filter(lambda p:p.requires_grad,model.parameters()), lr=0.0004)

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self,unfreeze_at_epch=1):
        self._unfreeze_at_epoch = unfreeze_at_epch
    def freeze_before_training(self,pl_module):
        self.freeze(pl_module.feature_extractor)
    def finetune_function(self,pl_module,current_epoch,optimizer,optimizer_idx):
        if current_epoch ==self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor,
                optimizer=optimizer,
                train_bn=True
            )






if __name__=="__main__":
    torch.multiprocessing.freeze_support()
    pd.set_option('mode.chained_assignment', None)
    pl.seed_everything(42)

    BATCH_SIZE = 1
    N_EPOCHS = 2
    data_module = BoolqDataModule(train_data, val_data, tokenizer, batch_size=BATCH_SIZE)
    data_module.setup()

    model = T5ForConditionalGeneration.from_pretrained(model_name,return_dict=True)
    model = BoolqModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=-1,
        verbose=True,
        monitor="val_loss",
        mode="min",

    )
    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=1,




    )

    trainer.fit(model, data_module)
    # trainer.save_checkpoint("final.ckpt")
    # sample_question = train_data.iloc[0]