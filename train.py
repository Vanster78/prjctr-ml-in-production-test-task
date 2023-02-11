from torch.nn.modules import MSELoss
from transformers import \
    AutoTokenizer,\
    AutoModelForSequenceClassification,\
    TrainingArguments,\
    Trainer
from dataset import CommonLitDataset


class RMSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        predictions = model(**inputs)['logits'][:, 0]
        loss = MSELoss()(predictions, inputs['labels']) ** 0.5
        if return_outputs:
            return loss, predictions
        else:
            return loss


MODEL = 'bert-base-cased'
DATA_DIR = 'data/train.csv'


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)

    train_dataset, val_dataset = CommonLitDataset.create(DATA_DIR, tokenizer)
    training_args = TrainingArguments(output_dir='training_results',
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch')
    trainer = RMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    train_metrics = trainer.evaluate(train_dataset)
    val_metrics = trainer.evaluate(val_dataset)
    print(train_metrics)
    print(val_metrics)


if __name__ == '__main__':
    main()
