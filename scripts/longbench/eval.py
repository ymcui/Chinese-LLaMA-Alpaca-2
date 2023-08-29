# The script is from https://github.com/THUDM/LongBench
import os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir')

args = parser.parse_args()

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "nq": qa_f1_score,
    "triviaqa": qa_f1_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    scores = dict()
    all_files = os.listdir(f"{args.pred_dir}/pred/")
    for filename in all_files:
        predictions, answers = [], []
        dataset = filename.split('.')[0]
        with open(f"{args.pred_dir}/pred/{filename}", "r") as f:
            print(filename)
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    with open(f"{args.pred_dir}/result.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
