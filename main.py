from fastapi import FastAPI
from pydantic import BaseModel, constr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = 'bert-base-cased'
MODEL_PATH = 'training_results/checkpoint-852'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
app = FastAPI()


class UserRequestIn(BaseModel):
    text: constr(min_length=1)


class ScoredLabelsOut(BaseModel):
    score: float


@app.post("/prediction", response_model=ScoredLabelsOut)
def read_classification(user_request_in: UserRequestIn):
    tokenized = tokenizer(user_request_in.text, return_tensors='pt')
    return {'score': model(**tokenized).logits[0, 0].item()}
