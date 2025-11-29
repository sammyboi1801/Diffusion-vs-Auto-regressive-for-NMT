
import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_calc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_bleu(reference, candidate):
    """
    Calculates BLEU score for a single reference and candidate sentence.
    :param reference: The reference sentence (string).
    :param candidate: The candidate sentence (string).
    :return: BLEU score.
    """
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge(reference, candidate):
    """
    Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    :param reference: The reference sentence (string).
    :param candidate: The candidate sentence (string).
    :return: A dictionary of ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {key: value.fmeasure for key, value in scores.items()}

def calculate_bert_score(reference, candidate):
    """
    Calculates BERTScore.
    :param reference: The reference sentence (string).
    :param candidate: The candidate sentence (string).
    :return: BERTScore F1.
    """
    _, _, f1 = bert_score_calc([candidate], [reference], lang="en", verbose=True)
    return f1.mean().item()

def evaluate_metrics(references, candidates, output_dir="tests/results"):
    """
    Calculates and logs BLEU, ROUGE, and BERTscore for lists of references and candidates.
    :param references: A list of reference sentences.
    :param candidates: A list of candidate sentences.
    :param output_dir: Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bleu_scores = []
    rouge_scores = []
    bert_scores = []

    for ref, cand in zip(references, candidates):
        bleu_scores.append({'reference': ref, 'candidate': cand, 'bleu_score': calculate_bleu(ref, cand)})
        
        rouge_results = calculate_rouge(ref, cand)
        rouge_scores.append({'reference': ref, 'candidate': cand, **rouge_results})
        
        bert_scores.append({'reference': ref, 'candidate': cand, 'bert_score': calculate_bert_score(ref, cand)})

    # Save scores to CSV
    pd.DataFrame(bleu_scores).to_csv(os.path.join(output_dir, "bleu_scores.csv"), index=False)
    logging.info(f"BLEU scores saved to {os.path.join(output_dir, 'bleu_scores.csv')}")

    pd.DataFrame(rouge_scores).to_csv(os.path.join(output_dir, "rouge_scores.csv"), index=False)
    logging.info(f"ROUGE scores saved to {os.path.join(output_dir, 'rouge_scores.csv')}")

    pd.DataFrame(bert_scores).to_csv(os.pathJ.join(output_dir, "bert_scores.csv"), index=False)
    logging.info(f"BERT scores saved to {os.path.join(output_dir, 'bert_scores.csv')}")


if __name__ == '__main__':
    # Example Usage
    sample_references = [
        "The cat sat on the mat.",
        "The quick brown fox jumps over the lazy dog."
    ]
    sample_candidates = [
        "A cat is on the mat.",
        "A fast brown fox leaps over a lazy dog."
    ]
    
    evaluate_metrics(sample_references, sample_candidates)
