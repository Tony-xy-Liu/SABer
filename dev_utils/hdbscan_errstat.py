import logging
import multiprocessing
from functools import reduce

import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def EArecruit(p):  # Error Analysis for all recruits per sag
    col_id, temp_id, temp_clust_df, temp_contig_df, temp_src2contig_list, temp_src2strain_list = p
    temp_clust_df['hdbscan'] = 1
    temp_contig_df[col_id] = temp_id
    df_list = [temp_contig_df, temp_clust_df]
    merge_recruits_df = reduce(lambda left, right: pd.merge(left, right,
                                                            on=[col_id, 'contig_id'],
                                                            how='left'),
                               df_list
                               )
    merge_recruits_df.fillna(-1, inplace=True)
    merge_recruits_df['exact_truth'] = [1 if x in temp_src2contig_list else -1
                                        for x in merge_recruits_df['contig_id']
                                        ]
    merge_recruits_df['strain_truth'] = [1 if x in temp_src2strain_list else -1
                                         for x in merge_recruits_df['contig_id']
                                         ]
    contig_id_list = list(merge_recruits_df['contig_id'])
    contig_bp_list = list(merge_recruits_df['bp_cnt'])
    exact_truth = list(merge_recruits_df['exact_truth'])
    strain_truth = list(merge_recruits_df['strain_truth'])
    algo_list = ['hdbscan']
    stats_lists = []
    for algo in algo_list:
        pred = list(merge_recruits_df[algo])
        stats_lists.extend(recruit_stats([temp_id, algo, contig_id_list, contig_bp_list,
                                          exact_truth, strain_truth, pred
                                          ]))
    return stats_lists


def recruit_stats(p):
    sag_id, algo, contig_id_list, contig_bp_list, exact_truth, strain_truth, pred = p
    pred_df = pd.DataFrame(zip(contig_id_list, contig_bp_list, pred),
                           columns=['contig_id', 'contig_bp', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['algorithm'] = algo
    pred_df = pred_df[['sag_id', 'algorithm', 'contig_id', 'contig_bp', 'pred']]
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth

    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    str_list = calc_stats(sag_id, 'strain', algo, TP, FP, TN, FN,
                          pred_df['truth_strain'], pred_df['pred']
                          )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    x_list = calc_stats(sag_id, 'exact', algo, TP, FP, TN, FN,
                        pred_df['truth'], pred_df['pred']
                        )
    cat_list = [str_list, x_list]

    return cat_list


def calc_tp(y_truth, y_pred, bp_cnt):
    tp_list = pd.Series([1 if ((x[0] == 1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    tp_bp_list = pd.Series([x[0] * x[1] for x in zip(tp_list, bp_cnt)])
    TP = tp_bp_list.sum()

    return TP


def calc_fp(y_truth, y_pred, bp_cnt):
    fp_list = pd.Series([1 if ((x[0] == -1) & (x[1] == 1)) else 0 for x in zip(y_truth, y_pred)])
    fp_bp_list = pd.Series([x[0] * x[1] for x in zip(fp_list, bp_cnt)])
    FP = fp_bp_list.sum()

    return FP


def calc_tn(y_truth, y_pred, bp_cnt):
    tn_list = pd.Series([1 if ((x[0] == -1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    tn_bp_list = pd.Series([x[0] * x[1] for x in zip(tn_list, bp_cnt)])
    TN = tn_bp_list.sum()

    return TN


def calc_fn(y_truth, y_pred, bp_cnt):
    fn_list = pd.Series([1 if ((x[0] == 1) & (x[1] == -1)) else 0 for x in zip(y_truth, y_pred)])
    fn_bp_list = pd.Series([x[0] * x[1] for x in zip(fn_list, bp_cnt)])
    FN = fn_bp_list.sum()

    return FN


def calc_stats(sag_id, level, algo, TP, FP, TN, FN, y_truth, y_pred):
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    D = ((S * P) * (1 - S) * (1 - P)) ** (1 / 2)
    if D == 0:
        D = 1
    MCC = ((TP / N) - S * P) / D
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    oc_precision, oc_recall, _ = precision_recall_curve(y_truth, y_pred)
    AUC = auc(oc_recall, oc_precision)
    stat_list = [sag_id, level, algo, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


# setup mapping to CAMI ref genomes
cluster_df = pd.read_csv(
    '/home/ryan/Desktop/test_NMF/minhash_features/'
    'CAMI_high_GoldStandardAssembly.leaf.mahalanobis_cleaned.tsv',
    sep='\t', header=0
)
cluster_df['supercluster'] = cluster_df['best_label'].astype(str) + '|' + \
                             cluster_df['best_clean_label'].astype(str)

cluster_trim_df = cluster_df.query('best_label != -1')

src2contig_df = pd.read_csv('/home/ryan/Desktop/test_NMF/src2contig_map.tsv', header=0, sep='\t')
# src2contig_df = src2contig_df[src2contig_df['CAMI_genomeID'].notna()
#                                ].rename(columns={'@@SEQUENCEID': 'contig_id'})
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})

clust2src_df = cluster_trim_df.merge(src2contig_df[['contig_id', 'CAMI_genomeID', 'strain', 'bp_cnt']],
                                     on='contig_id', how='left')

# Add taxonomy to each cluster
clust_tax = []
for clust in clust2src_df['best_label'].unique():
    sub_clust_df = clust2src_df.query('best_label == @clust')
    exact_df = sub_clust_df.groupby(['CAMI_genomeID'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID']
    if not ex_label_df.empty:
        exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID'].values[0]
        strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()]['strain'].values[0]
        clust_tax.append([clust, exact_label, strain_label])

clust_tax_df = pd.DataFrame(clust_tax, columns=['best_label', 'exact_label', 'strain_label'])
clust2label_df = clust_tax_df.merge(cluster_trim_df, on='best_label', how='left')
clust2contig_df = clust2label_df[['best_label', 'contig_id', 'exact_label', 'strain_label']].drop_duplicates()
contig_bp_df = src2contig_df[['contig_id', 'bp_cnt']]
# setup multithreading pool
nthreads = 8
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(clust2contig_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = clust2contig_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sub_clust_df['exact_label'].values[0]
    strain_id = sub_clust_df['strain_label'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list, src2strain_list])

results = pool.imap_unordered(EArecruit, arg_list)
score_list = []
for i, output in tqdm(enumerate(results, 1)):
    score_list.extend(output)
logging.info('\n')
pool.close()
pool.join()
score_df = pd.DataFrame(score_list, columns=['best_label', 'level', 'algorithm',
                                             'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                             'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                             ])
print(score_df.head())
sort_score_df = score_df.sort_values(['best_label', 'level', 'precision', 'sensitivity'],
                                     ascending=[False, False, True, True]
                                     )
score_tax_df = sort_score_df.merge(clust_tax_df, on='best_label', how='left')
score_tax_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                    'umap.best_label.100.100.leaf.errstat.tsv', index=False, sep='\t')
score_tax_df = sort_score_df.groupby(['level', 'algorithm'])[
    ['precision', 'sensitivity', 'MCC', 'AUC', 'F1']].mean().reset_index()
score_tax_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                    'umap.best_label.100.100.leaf.errstat.mean.tsv', index=False, sep='\t')

#####################################################################################################################

# Add taxonomy to each cluster
clust_tax = []
for clust in clust2src_df['supercluster'].unique():
    sub_clust_df = clust2src_df.query('supercluster == @clust')
    exact_df = sub_clust_df.groupby(['CAMI_genomeID'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID']
    if not ex_label_df.empty:
        exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID'].values[0]
        strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()]['strain'].values[0]
        clust_tax.append([clust, exact_label, strain_label])

clust_tax_df = pd.DataFrame(clust_tax, columns=['supercluster', 'exact_label', 'strain_label'])
clust2label_df = clust_tax_df.merge(cluster_trim_df, on='supercluster', how='left')
clust2contig_df = clust2label_df[['supercluster', 'contig_id', 'exact_label', 'strain_label']].drop_duplicates()
contig_bp_df = src2contig_df[['contig_id', 'bp_cnt']]
# setup multithreading pool
nthreads = 8
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(clust2contig_df['supercluster'].unique()):
    # subset recruit dataframes
    sub_clust_df = clust2contig_df.query('supercluster == @clust')
    dedup_clust_df = sub_clust_df[['supercluster', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sub_clust_df['exact_label'].values[0]
    strain_id = sub_clust_df['strain_label'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['supercluster', clust, dedup_clust_df, contig_bp_df, src2contig_list, src2strain_list])

results = pool.imap_unordered(EArecruit, arg_list)
score_list = []
for i, output in tqdm(enumerate(results, 1)):
    score_list.extend(output)
logging.info('\n')
pool.close()
pool.join()
score_df = pd.DataFrame(score_list, columns=['supercluster', 'level', 'algorithm',
                                             'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                             'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                             ])
print(score_df.head())
sort_score_df = score_df.sort_values(['supercluster', 'level', 'precision', 'sensitivity'],
                                     ascending=[False, False, True, True]
                                     )
score_tax_df = sort_score_df.merge(clust_tax_df, on='supercluster', how='left')
score_tax_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                    'umap.clean_cluster.mahalanobis.leaf.errstat.tsv', index=False, sep='\t')
score_tax_df = sort_score_df.groupby(['level', 'algorithm'])[
    ['precision', 'sensitivity', 'MCC', 'AUC', 'F1']].mean().reset_index()
score_tax_df.to_csv('/home/ryan/Desktop/test_NMF/minhash_features/'
                    'umap.clean_cluster.mahalanobis.leaf.errstat.mean.tsv', index=False, sep='\t')
