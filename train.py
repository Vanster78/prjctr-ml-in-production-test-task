from torch.nn.modules import MSELoss
from transformers import \
    AutoTokenizer,\
    AutoModelForSequenceClassification,\
    TrainingArguments,\
    Trainer

from dataset import CommonLitDataset


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        predictions = model(**inputs)['logits']
        loss = MSELoss()(predictions, inputs['labels'])
        if return_outputs:
            return loss, predictions
        else:
            return loss


def compute_rmse(eval_pred):
    predictions, labels = eval_pred
    rmse = MSELoss()(predictions, predictions) ** 0.5
    return {"rmse": rmse}


MODEL = 'bert-base-cased'
DATA_DIR = 'data'
DEVICE = 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)

train_dataset, val_dataset = CommonLitDataset.create(DATA_DIR,
                                                     tokenizer,
                                                     train=True)
train_dataset = train_dataset
test_dataset = CommonLitDataset.create(DATA_DIR, tokenizer, train=False)

training_args = TrainingArguments(output_dir='training_results',
                                  evaluation_strategy='epoch')
trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_rmse,
)
trainer.train()
