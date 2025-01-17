#!/usr/bin/env python

import sys

import pandas as pd
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

pd.options.mode.chained_assignment = None  # default='warn'


def recruitSubs(p):
    sag_id, mh_sag_df, nmf_rec, tetra_table, cov_table, gamma, nu, \
    src2contig_list, src2strain_list, contig_bp_df = p

    # load nmf recruits file
    nmf_df = pd.read_csv(nmf_rec, sep='\t', header=0)
    nmf_rec_df = nmf_df.loc[nmf_df['sag_id'] == sag_id]  # this is the testing subset
    # load nmf feature file
    tetra_feat_df = pd.read_csv(tetra_table, sep='\t', header=0)
    tetra_feat_df['subcontig_id'] = tetra_feat_df['contig_id']
    tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_feat_df['contig_id']]
    # load covm file
    cov_df = pd.read_csv(cov_table, sep='\t', header=0)
    cov_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
    cov_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_df['subcontig_id']]
    cov_df.set_index('subcontig_id', inplace=True)

    # subset all tables before merging
    sag_tetra_df = tetra_feat_df.loc[tetra_feat_df['contig_id'].isin(mh_sag_df['contig_id'])]
    mg_tetra_df = tetra_feat_df.loc[tetra_feat_df['contig_id'].isin(nmf_rec_df['contig_id'])]
    sag_tetra_df.drop(columns=['contig_id'], inplace=True)
    mg_tetra_df.drop(columns=['contig_id'], inplace=True)
    sag_tetra_df.set_index('subcontig_id', inplace=True)
    mg_tetra_df.set_index('subcontig_id', inplace=True)
    sag_cov_df = cov_df.loc[cov_df['contig_id'].isin(mh_sag_df['contig_id'])]
    mg_cov_df = cov_df.loc[cov_df['contig_id'].isin(nmf_rec_df['contig_id'])]
    sag_cov_df.drop(columns=['contig_id'], inplace=True)
    mg_cov_df.drop(columns=['contig_id'], inplace=True)

    # merge covM and NMF
    sag_join_df = sag_tetra_df.join(sag_cov_df, lsuffix='_tetra', rsuffix='_covm')
    mg_join_df = mg_tetra_df.join(mg_cov_df, lsuffix='_tetra', rsuffix='_covm')

    # start ocsvm cross validation analysis
    final_pass_df = runOCSVM(sag_join_df, mg_join_df, sag_id, gamma, nu)
    sub_tetra_df = tetra_feat_df.loc[~tetra_feat_df['contig_id'].isin(mh_sag_df['contig_id'])]
    complete_df = pd.DataFrame(list(sub_tetra_df['subcontig_id']), columns=['subcontig_id'])
    complete_df['sag_id'] = sag_id
    complete_df['nu'] = nu
    complete_df['gamma'] = gamma
    complete_df['contig_id'] = [x.rsplit('_', 1)[0] for x in sub_tetra_df['subcontig_id']]
    merge_recruits_df = pd.merge(complete_df, final_pass_df,
                                 on=['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id'],
                                 how='outer'
                                 )
    merge_recruits_df.fillna(-1, inplace=True)
    merge_recruits_df['exact_truth'] = [1 if x in src2contig_list else -1
                                        for x in merge_recruits_df['contig_id']
                                        ]
    merge_recruits_df['strain_truth'] = [1 if x in src2strain_list else -1
                                         for x in merge_recruits_df['contig_id']
                                         ]
    merge_bp_df = merge_recruits_df.merge(contig_bp_df, on='contig_id', how='left')
    subcontig_id_list = list(merge_bp_df['subcontig_id'])
    contig_id_list = list(merge_bp_df['contig_id'])
    contig_bp_list = list(merge_bp_df['bp_cnt'])
    exact_truth = list(merge_bp_df['exact_truth'])
    strain_truth = list(merge_bp_df['strain_truth'])
    pred = list(merge_bp_df['pred'])

    stats_lists = recruit_stats([sag_id, gamma, nu, subcontig_id_list, contig_id_list, contig_bp_list,
                                 exact_truth, strain_truth, pred
                                 ])
    return stats_lists


def runOCSVM(sag_df, mg_df, sag_id, gamma, nu):
    # fit OCSVM
    clf = svm.OneClassSVM(nu=nu, gamma=gamma)
    clf.fit(sag_df.values)
    mg_pred = clf.predict(mg_df.values)
    contig_id_list = [x.rsplit('_', 1)[0] for x in mg_df.index.values]
    pred_df = pd.DataFrame(zip(mg_df.index.values, contig_id_list, mg_pred),
                           columns=['subcontig_id', 'contig_id', 'pred']
                           )
    pred_df['nu'] = nu
    pred_df['gamma'] = gamma
    pred_df['sag_id'] = sag_id
    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'pred']]

    return pred_df


def recruit_stats(p):
    sag_id, gam, n, subcontig_id_list, contig_id_list, contig_bp_list, exact_truth, strain_truth, pred = p
    pred_df = pd.DataFrame(zip(subcontig_id_list, contig_id_list, contig_bp_list, pred),
                           columns=['subcontig_id', 'contig_id', 'contig_bp', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['gamma'] = gam
    pred_df['nu'] = n

    pred_df = pred_df[['sag_id', 'nu', 'gamma', 'subcontig_id', 'contig_id', 'contig_bp', 'pred']]

    val_perc = pred_df.groupby('contig_id')['pred'].value_counts(
        normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc['pred'] == 1]
    major_df = pos_perc.loc[pos_perc['precent'] >= 0.51]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                  for x in pred_df['contig_id']
                  ]
    pos_pred_list = list(set(pred_df.loc[pred_df['pred'] == 1]['contig_id']))
    all_pred = [1 if x in pos_pred_list else -1
                for x in pred_df['contig_id']
                ]
    pred_df['all_pred'] = all_pred
    pred_df['major_pred'] = major_pred
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth
    dedup_pred_df = pred_df.drop_duplicates(subset=['sag_id', 'nu', 'gamma', 'contig_id', 'contig_bp',
                                                    'pred', 'all_pred', 'major_pred', 'truth',
                                                    'truth_strain'])
    # ALL Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth_strain'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    all_str_list = calc_stats(sag_id, 'strain', 'all', gam, n, TP, FP, TN, FN,
                              dedup_pred_df['truth_strain'], dedup_pred_df['all_pred']
                              )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['all_pred'], dedup_pred_df['contig_bp'])
    all_x_list = calc_stats(sag_id, 'exact', 'all', gam, n, TP, FP, TN, FN,
                            dedup_pred_df['truth'], dedup_pred_df['all_pred']
                            )

    # Majority-Rule Recruits
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth_strain'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    maj_str_list = calc_stats(sag_id, 'strain', 'majority', gam, n, TP, FP, TN, FN,
                              dedup_pred_df['truth_strain'], dedup_pred_df['major_pred']
                              )
    # Majority-Rule Recruits
    # calculate for exact-level match
    TP = calc_tp(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FP = calc_fp(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    TN = calc_tn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    FN = calc_fn(dedup_pred_df['truth'], dedup_pred_df['major_pred'], dedup_pred_df['contig_bp'])
    maj_x_list = calc_stats(sag_id, 'exact', 'majority', gam, n, TP, FP, TN, FN,
                            dedup_pred_df['truth'], dedup_pred_df['major_pred']
                            )
    filter_pred_df = dedup_pred_df.loc[dedup_pred_df['major_pred'] == 1]

    return all_str_list, all_x_list, maj_str_list, maj_x_list, filter_pred_df


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


def calc_stats(sag_id, level, include, gam, n, TP, FP, TN, FN, y_truth, y_pred):
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
    stat_list = [sag_id, level, include, gam, n, precision, sensitivity, MCC, AUC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list



# Build final table for testing
minhash_recruits = sys.argv[1]
nmf_recruits = sys.argv[2]
tetra_dat = sys.argv[3]
cov_dat = sys.argv[4]


# load minhash file
minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
# load nmf recruits file
nmf_df = pd.read_csv(nmf_recruits, sep='\t', header=0)
# load nmf feature file
tetra_feat_df = pd.read_csv(tetra_dat, sep='\t', header=0)
tetra_feat_df['subcontig_id'] = tetra_feat_df['contig_id']
tetra_feat_df['contig_id'] = [x.rsplit('_', 1)[0] for x in tetra_feat_df['contig_id']]
# load covm file
cov_df = pd.read_csv(cov_dat, sep='\t', header=0)
cov_df.rename(columns={'contigName': 'subcontig_id'}, inplace=True)
cov_df['contig_id'] = [x.rsplit('_', 1)[0] for x in cov_df['subcontig_id']]
cov_df.set_index('subcontig_id', inplace=True)

pred_df_list = []
for sag_id in minhash_df['sag_id'].unique():
    print(sag_id)
    # subset all tables before merging
    mh_sag_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
    nmf_rec_df = nmf_df.loc[nmf_df['sag_id'] == sag_id]  # this is the testing subset
    sag_tetra_df = tetra_feat_df.loc[tetra_feat_df['contig_id'].isin(mh_sag_df['contig_id'])]
    mg_tetra_df = tetra_feat_df.loc[tetra_feat_df['contig_id'].isin(nmf_rec_df['contig_id'])]
    sag_tetra_df.drop(columns=['contig_id'], inplace=True)
    mg_tetra_df.drop(columns=['contig_id'], inplace=True)
    sag_tetra_df.set_index('subcontig_id', inplace=True)
    mg_tetra_df.set_index('subcontig_id', inplace=True)
    sag_cov_df = cov_df.loc[cov_df['contig_id'].isin(mh_sag_df['contig_id'])]
    mg_cov_df = cov_df.loc[cov_df['contig_id'].isin(nmf_rec_df['contig_id'])]
    sag_cov_df.drop(columns=['contig_id'], inplace=True)
    mg_cov_df.drop(columns=['contig_id'], inplace=True)

    # merge covM and NMF
    sag_join_df = sag_tetra_df.join(sag_cov_df, lsuffix='_tetra', rsuffix='_covm')
    mg_join_df = mg_tetra_df.join(mg_cov_df, lsuffix='_tetra', rsuffix='_covm')

    # start ocsvm cross validation analysis
    pred_df = runOCSVM(sag_join_df, mg_join_df, sag_id, 0.1, 0.1)
    val_perc = pred_df.groupby('contig_id')['pred'].value_counts(
        normalize=True).reset_index(name='precent')
    pos_perc = val_perc.loc[val_perc['pred'] == 1]
    major_df = pos_perc.loc[pos_perc['precent'] != 0]
    major_pred = [1 if x in list(major_df['contig_id']) else -1
                  for x in pred_df['contig_id']
                  ]
    pred_df['major_pred'] = major_pred
    pred_filter_df = pred_df.loc[pred_df['major_pred'] == 1]
    merge_df = pd.concat([mh_sag_df[['sag_id', 'contig_id']],
                          pred_filter_df[['sag_id', 'contig_id']]]
                         ).drop_duplicates()
    pred_df_list.append(merge_df)
    print('Recruited', pred_filter_df.shape[0], 'subcontigs...')
    print('Total of', pred_filter_df[['sag_id', 'contig_id']].drop_duplicates().shape[0],
          'contigs...')
    print('Total of', merge_df.shape[0], 'contigs with minhash...')

final_pred_df = pd.concat(pred_df_list)
final_pred_df.to_csv('~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.allfeat_recruits.tsv',
                     sep='\t', index=False
                     )

sys.exit()

# Below is to run cross validation for all features table
#################################################
# Inputs
#################################################
sag_id = sys.argv[1]
minhash_recruits = sys.argv[2]
nmf_recruits = sys.argv[3]
tetra_dat = sys.argv[4]
cov_dat = sys.argv[5]
nmf_output = sys.argv[6]
best_output = sys.argv[7]
src2contig_file = sys.argv[8]
sag2cami_file = sys.argv[9]
subcontig_file = sys.argv[10]
nthreads = int(sys.argv[11])

# Example:
# python
# dev_utils/test_NMF.py
# 1021_F_run134.final.scaffolds.gt1kb.2806
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.201.mhr_trimmed_recruits.tsv
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.abr_trimmed_recruits.tsv
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.nmf_trans_20.tsv
# ~/Desktop/test_NMF/CAMI_high_GoldStandardAssembly.covM.scaled.tsv
# ~/Desktop/test_NMF/all_preds/1021_F_run134.final.scaffolds.gt1kb.2806.all_scores.tsv
# ~/Desktop/test_NMF/all_preds/1021_F_run134.final.scaffolds.gt1kb.2806.all_best.tsv
# ~/Desktop/test_NMF/src2sag_map.tsv
# 6

#################################################


# setup mapping to CAMI ref genomes
minhash_df = pd.read_csv(minhash_recruits, sep='\t', header=0)
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df[src2contig_df['CAMI_genomeID'].notna()]
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
subcontig_df = pd.read_csv(subcontig_file, sep='\t', header=0)
contig_df = subcontig_df.drop(['subcontig_id'], axis=1).drop_duplicates()
contig_bp_df = contig_df.merge(src2contig_df[['@@SEQUENCEID', 'bp_cnt']].rename(
    columns={'@@SEQUENCEID': 'contig_id'}), on='contig_id', how='left'
)

'''
# builds the sag to cami ID mapping file
mh_list = list(minhash_df['sag_id'].unique())
cami_list = [str(x) for x in src2contig_df['CAMI_genomeID'].unique()]
sag2cami_list = []
print('Mapping Sources to Synthetic SAGs...')
for sag_id in mh_list:
    match = difflib.get_close_matches(str(sag_id), cami_list, n=1, cutoff=0)[0]
    m_len = len(match)
    sub_sag_id = sag_id[:m_len]
    if sub_sag_id != match:
        match = difflib.get_close_matches(str(sub_sag_id), cami_list, n=1, cutoff=0)[0]
        if match == sub_sag_id:
            print("PASSED:", sag_id, sub_sag_id, match)
        else:
            m1_len = len(match)
            sub_sag_id = sag_id[:m_len]
            sub_sub_id = sub_sag_id[:m1_len].split('.')[0]
            match = difflib.get_close_matches(str(sub_sub_id), cami_list, n=1, cutoff=0)[0]
    sag2cami_list.append([sag_id, match])
sag2cami_df = pd.DataFrame(sag2cami_list, columns=['sag_id', 'CAMI_genomeID'])
sag2cami_df.to_csv("~/Desktop/test_NMF/sag2cami_map.tsv", index=False, sep='\t')
'''

# Run GridCV on the [All Hz + Cov] features
sag_mh_df = minhash_df.loc[minhash_df['sag_id'] == sag_id]
if sag_mh_df.shape[0] != 0:
    # Map Sources/SAGs to Strain IDs
    src_id = list(sag2cami_df.loc[sag2cami_df['sag_id'] == sag_id]['CAMI_genomeID'])[0]
    strain_id = list(src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id
                                       ]['strain'])[0]
    src_sub_df = src2contig_df.loc[src2contig_df['CAMI_genomeID'] == src_id]
    strain_sub_df = src2contig_df.loc[src2contig_df['strain'] == strain_id]
    src2contig_list = list(set(src_sub_df['@@SEQUENCEID'].values))
    src2strain_list = list(set(strain_sub_df['@@SEQUENCEID'].values))
    print(sag_id, src_id, strain_id)

    gamma_range = [10 ** k for k in range(-6, 6)]
    gamma_range.extend(['scale'])
    nu_range = [k / 10 for k in range(1, 10, 1)]

    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for gam in gamma_range:
        for n in nu_range:
            arg_list.append([sag_id, sag_mh_df, nmf_recruits, tetra_dat, cov_dat,
                             gam, n, src2contig_list, src2strain_list, contig_bp_df
                             ])
    results = pool.imap_unordered(recruitSubs, arg_list)
    score_list = []
    for i, output in enumerate(results, 1):
        print('\rRecruiting with All Tetra Model: {}/{}'.format(i, len(arg_list)))
        score_list.append(output[0])
        score_list.append(output[1])
        score_list.append(output[2])
        score_list.append(output[3])
    logging.info('\n')
    pool.close()
    pool.join()
    score_df = pd.DataFrame(score_list, columns=['sag_id', 'level', 'inclusion', 'gamma', 'nu',
                                                 'precision', 'sensitivity', 'MCC', 'AUC', 'F1',
                                                 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN'
                                                 ])
    score_df.to_csv(nmf_output, index=False, sep='\t')
    sort_score_df = score_df.sort_values(['MCC'], ascending=[False])
    best_MCC = sort_score_df['MCC'].iloc[0]
    best_df = score_df.loc[score_df['MCC'] == best_MCC]
    best_df.to_csv(best_output, index=False, sep='\t')
else:
    print(sag_id, ' has no minhash recruits...')
