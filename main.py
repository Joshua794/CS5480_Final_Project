import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#data
def load_data(path):
    df = pd.read_csv(path)

    # Adjust column names if needed
    questions = df["question"].tolist()
    answers = df["short_answers"].tolist()

    return questions, answers


#model init
def load_model():
    model_name = "google/flan-t5-small" #can be any model, like flan-t5-base, but I don't have a beefy computer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


#answer question
def generate_answer(question, tokenizer, model):
    prompt = f"Answer the question: {question}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=64)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


#eval criteria
def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())


def f1_score(pred, truth):
    pred_tokens = pred.lower().split()
    truth_tokens = truth.lower().split()

    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)

    return 2 * (precision * recall) / (precision + recall)


#run baseline experiment
def run_baseline(questions, answers, tokenizer, model, limit=100):
    em_scores = []
    f1_scores = []

    for i in range(min(limit, len(questions))):
        q = questions[i]
        a = answers[i]

        pred = generate_answer(q, tokenizer, model)

        em = exact_match(pred, a)
        f1 = f1_score(pred, a)

        em_scores.append(em)
        f1_scores.append(f1)

        if i % 10 == 0:
            print(f"[{i}] Q: {q}")
            print(f"Pred: {pred}")
            print(f"Truth: {a}")
            print("-" * 40)

    return np.mean(em_scores), np.mean(f1_scores)

def main():
    data_path = "Natural-Questions-Filtered.csv"

    print("Loading data...")
    questions, answers = load_data(data_path)

    print("Loading model...")
    tokenizer, model = load_model()

    print("Running baseline...")
    em, f1 = run_baseline(questions, answers, tokenizer, model)

    print("\n=== RESULTS ===")
    print(f"Exact Match: {em:.4f}")
    print(f"F1 Score:    {f1:.4f}")


if __name__ == "__main__":
    main()