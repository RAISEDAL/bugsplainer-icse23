import os.path
import sys
from typing import Union

import numpy as np
import torch
from pandas.io.parsers.readers import read_csv
from pandas.core.frame import DataFrame
from sentence_transformers import SentenceTransformer

from src.bugsplainer.smooth_bleu import bleuFromMaps


class ScoreCalculator:
    def __init__(self, source: Union[str, DataFrame]):
        if type(source) is str:
            self.explanation_df: DataFrame = read_csv(source, keep_default_na=False)
        else:
            self.explanation_df: DataFrame = source
        self.num = len(self.explanation_df)
        self.ref_map = {
            i: [self.explanation_df['gold'].iloc[i]]
            for i in range(self.num)
        }

        self.sbert = SentenceTransformer('stsb-roberta-large', device='cuda')
        self.ref_tensor = self.sbert.encode(self.explanation_df['gold'].tolist(), convert_to_tensor=True)

    def compute_bleu(self, col: str):
        gen_map = {
            i: [self.explanation_df[col].iloc[i]]
            for i in range(self.num)
        }

        return bleuFromMaps(self.ref_map, gen_map)[0]

    def compute_semantic_similarity(self, col: str):
        gen_tensor = self.sbert.encode(self.explanation_df[col].tolist(), convert_to_tensor=True)
        cosine_score = torch.cosine_similarity(self.ref_tensor, gen_tensor, dim=1)
        assert cosine_score.shape == (self.num,), cosine_score.shape
        return torch.mean(cosine_score).item() * 100

    def compute_exact_match(self, col: str):
        return (self.explanation_df[col] == self.explanation_df['gold']).mean() * 100


def compute_scores_from_gen(gen_dir: str):
    result_dir = os.path.join('runs', gen_dir, 'result')
    gold_df = read_csv(os.path.join(result_dir, 'test_best-bleu.gold'), sep='\t', names=['id', 'gold'])
    gen_df = read_csv(os.path.join(result_dir, 'test_best-bleu.output'), sep='\t', names=['id', 'gen'])
    df = gold_df.merge(gen_df, on='id')
    score_calculator = ScoreCalculator(df)

    print(f'BLEU score: {score_calculator.compute_bleu("gen")}')
    print(f'Semantic Similarity: {score_calculator.compute_semantic_similarity("gen")}')
    print(f'Exact Match: {score_calculator.compute_exact_match("gen")}')


def compute_pyflakes_score_from_sample(source_file: str):
    explanation_df: DataFrame = read_csv(source_file, keep_default_na=False)
    pyflakes_df = read_csv(
        'output/run4/all_pyflakes_error.csv',
        usecols=['fix_commit_sha', 'pyflakes_error', 'repo'],
        keep_default_na=False,
    )
    result_with_pyflakes_df: DataFrame = explanation_df.merge(pyflakes_df, 'left', on=['fix_commit_sha', 'repo'])
    result_with_pyflakes_df['pyflakes_error'] = result_with_pyflakes_df['pyflakes_error'].fillna('')

    unique_groups = result_with_pyflakes_df.groupby(['fix_commit_sha', 'repo'])
    bleu, sem_sim, em = [], [], []

    sbert = SentenceTransformer('stsb-roberta-large', device='cuda')

    for _id, group in unique_groups:
        em.append((group['pyflakes_error'] == group['gold']).any())

        gen_map = {
            i: [group['pyflakes_error'].iloc[i]]
            for i in range(len(group))
        }
        ref_map = {
            i: [group['gold'].iloc[i]]
            for i in range(len(group))
        }
        bleu_scores = bleuFromMaps(ref_map, gen_map)
        bleu.append(max(*bleu_scores))

        gen_tensor = sbert.encode(group['pyflakes_error'].tolist(), convert_to_tensor=True)
        ref_tensor = sbert.encode(group['gold'].tolist(), convert_to_tensor=True)
        cosine_score = torch.cosine_similarity(ref_tensor, gen_tensor, dim=1)
        assert cosine_score.shape == (len(group),), cosine_score.shape
        sem_sim.append(torch.max(cosine_score).item() * 100)

    print(f'BLEU score: {np.mean(bleu)}')
    print(f'Semantic Similarity: {np.mean(sem_sim)}')
    print(f'Exact Match: {np.mean(em)}')


def main():
    if len(sys.argv) == 1:
        compute_scores_from_sample()
    elif sys.argv[1].lower() == 'pyflakes':
        compute_pyflakes_score_from_sample(explanations_csv)
    else:
        compute_scores_from_gen(sys.argv[1])


def compute_scores_from_sample():
    score_calculator = ScoreCalculator(explanations_csv)
    columns = ['60M', '220M', 'CodeT5', 'nngen', 'commitgen', 'pyflakes']
    bleu_scores = {
        col: score_calculator.compute_bleu(col)
        for col in columns
    }
    semantic_similarities = {
        col: score_calculator.compute_semantic_similarity(col)
        for col in columns
    }
    exact_match_scores = {
        col: score_calculator.compute_exact_match(col)
        for col in columns
    }
    score_df = DataFrame({
        'BLEU': bleu_scores,
        'SemanticSimilarity': semantic_similarities,
        'ExactMatch': exact_match_scores,
    })
    print(score_df)
    score_df.to_csv('output/run4/score.csv')


if __name__ == '__main__':
    explanations_csv = 'output/run4/generated_explanations.csv'
    main()
