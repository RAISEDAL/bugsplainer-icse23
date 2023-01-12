#%%
from pandas.io.parsers.readers import read_csv
from pandas.core.frame import DataFrame

#%%

score_df = read_csv('output/scores_of_5_runs.csv', index_col=[0, 1])
print('Scores of all 5 runs:')
print(score_df)
#%%

model_names = score_df.index.unique(level=0)
metric_names = score_df.index.unique(level=1)
mean_score_df = DataFrame(
    data=score_df.mean(axis=1).values.reshape(len(model_names), len(metric_names)),
    index=model_names,
    columns=metric_names,
)
print('Mean scores:')
print(mean_score_df)

#%%

from scipy.stats import wilcoxon, ttest_rel, mannwhitneyu
from cliffs_delta import cliffs_delta

wilcoxon_p_values = {
    metric: wilcoxon(score_df.loc['60M', metric], score_df.loc['CodeT5', metric], alternative='greater').pvalue
    for metric in metric_names
}
mannwhitneyu_p_values = {
    metric: mannwhitneyu(score_df.loc['60M', metric], score_df.loc['CodeT5', metric]).pvalue
    for metric in metric_names
}
ttest_p_values = {
    metric: ttest_rel(score_df.loc['60M', metric], score_df.loc['CodeT5', metric], alternative='greater').pvalue
    for metric in metric_names
}
cliffs_delta_values = {
    metric: cliffs_delta(score_df.loc['60M', metric], score_df.loc['CodeT5', metric])
    for metric in metric_names
}

print('Paired test between Bugsplainer and Fine-tuned CodeT5')
print(DataFrame({'wilcoxon': wilcoxon_p_values, 'mannwhitneyu': mannwhitneyu_p_values, 'ttest': ttest_p_values, 'cliffs_delta': cliffs_delta_values}))
