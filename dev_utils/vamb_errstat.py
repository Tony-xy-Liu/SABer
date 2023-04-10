#!/usr/bin/env python

import glob
import logging
import multiprocessing
import sys
from functools import reduce
from os import makedirs, path
from os.path import join as joinpath

import pandas as pd
import pyfastx
from tqdm import tqdm




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)






def EArecruit(p):  # Error Analysis for all recruits per sag
    col_id, temp_id, samp_id, temp_clust_df, temp_contig_df, temp_src2contig_list, \
    temp_src2strain_list, algorithm, src_id, strain_id, tot_bp_dict = p
    temp_clust_df[algorithm] = 1
    temp_contig_df[col_id] = temp_id
    temp_contig_df['sample_id'] = samp_id
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
    src_total_bp = tot_bp_dict[src_id]
    algo_list = [algorithm]
    stats_lists = []
    for algo in algo_list:
        pred = list(merge_recruits_df[algo])
        rec_stats_list = recruit_stats([temp_id, samp_id, algo, contig_id_list,
                                        contig_bp_list, exact_truth,
                                        strain_truth, pred, src_total_bp,
                                        src_id, strain_id
                                        ])
        stats_lists.extend(rec_stats_list)

    return stats_lists


def recruit_stats(p):
    sag_id, samp_id, algo, contig_id_list, contig_bp_list, \
    exact_truth, strain_truth, pred, tot_bp, src_id, strain_id = p
    pred_df = pd.DataFrame(zip(contig_id_list, contig_bp_list, pred, ),
                           columns=['contig_id', 'contig_bp', 'pred']
                           )
    pred_df['sag_id'] = sag_id
    pred_df['sample_id'] = samp_id
    pred_df['algorithm'] = algo
    pred_df = pred_df[['sag_id', 'sample_id', 'algorithm', 'contig_id', 'contig_bp', 'pred']]
    pred_df['truth'] = exact_truth
    pred_df['truth_strain'] = strain_truth
    # calculate for hybrid exact/strain-level matches
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth_strain'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    # compute total possible bp for each genome
    str_tot_bp_poss = TP + FN
    # Complete SRC genome is not always present in contigs, need to correct for that.
    working_bp = tot_bp - TP - FN
    corrected_FN = FN + working_bp
    str_list = calc_stats(sag_id, samp_id, 'strain_assembly', algo, TP, FP, TN, FN,
                          pred_df['truth_strain'], pred_df['pred']
                          )
    corr_str_list = calc_stats(sag_id, samp_id, 'strain_absolute', algo, TP, FP, TN, corrected_FN,
                               pred_df['truth_strain'], pred_df['pred']
                               )
    # ALL Recruits
    # calculate for exact-level match
    TP = calc_tp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FP = calc_fp(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    TN = calc_tn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    FN = calc_fn(pred_df['truth'], pred_df['pred'], pred_df['contig_bp'])
    # compute total possible bp for each genome
    exa_tot_bp_poss = TP + FN
    # Complete SRC genome is not always present in contigs, need to correct for that.
    working_bp = tot_bp - TP - FN
    corrected_FN = FN + working_bp
    x_list = calc_stats(sag_id, samp_id, 'exact_assembly', algo, TP, FP, TN, FN,
                        pred_df['truth'], pred_df['pred']
                        )
    corr_x_list = calc_stats(sag_id, samp_id, 'exact_absolute', algo, TP, FP, TN, corrected_FN,
                             pred_df['truth'], pred_df['pred']
                             )

    # Add total possible bp's for complete genome
    str_list.extend([str_tot_bp_poss, tot_bp])
    x_list.extend([exa_tot_bp_poss, tot_bp])
    corr_str_list.extend([str_tot_bp_poss, tot_bp])
    corr_x_list.extend([exa_tot_bp_poss, tot_bp])

    cat_list = [str_list, corr_str_list, x_list, corr_x_list]

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


def calc_stats(sag_id, samp_id, level, algo, TP, FP, TN, FN, y_truth, y_pred):
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    D = ((S * P) * (1 - S) * (1 - P)) ** (1 / 2)
    if D == 0:
        D = 1
    MCC = ((TP / N) - S * P) / D
    if precision + sensitivity == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    # oc_precision, oc_recall, _ = precision_recall_curve(y_truth, y_pred)
    # AUC = auc(oc_recall, oc_precision)
    stat_list = [sag_id, samp_id, level, algo, precision, sensitivity, MCC, F1,
                 N, S, P, TP, FP, TN, FN
                 ]

    return stat_list


def cnt_contig_bp(fasta_file):
    # counts basepairs/read contained in file
    # returns dictionary of {read_header:bp_count}

    fasta_records = get_seqs(fasta_file)
    fa_cnt_dict = {}
    for f_rec in fasta_records:
        fa_cnt_dict[f_rec[0]] = len(f_rec[1])
    return fa_cnt_dict


def cnt_total_bp(fasta_file):
    # counts total basepairs contained in file
    # returns fasta_file name and total counts for entire fasta file

    fasta_records = get_seqs(fasta_file)
    bp_sum = 0
    for f_rec in fasta_records:
        bp_sum += len(f_rec[1])
    return fasta_file, bp_sum


def get_seqs(fasta_file):
    fasta = pyfastx.Fasta(fasta_file, build_index=False)
    return fasta


def cluster2taxonomy(p):
    clust, clust2src_df = p
    sub_clust_df = clust2src_df.query('best_label == @clust')
    exact_df = sub_clust_df.groupby(['exact_label'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['exact_label']
    try:
        if not ex_label_df.empty:
            exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()
                                   ]['exact_label'].values[0]
            strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()
                                     ]['strain'].values[0]
            return [clust, exact_label, strain_label]
    except:
        print(sub_clust_df.head())
        sys.exit()


def runErrorAnalysis(bin_path, synsrc_path, src_metag_file,
                     sample_type, nthreads
                     ):
    ##################################################################################################
    # INPUT files
    sag_tax_map = joinpath('/home/ryan/SABer_bench/GenQC', 'CAMI2.gen2ncbi.csv') # synsrc_path, 'genome_taxa_info.tsv')
    mg_contig_map = joinpath(synsrc_path, 'gsa_mapping_pool.binning')
    src_contig_cnt = joinpath(synsrc_path, 'src_fasta.stats.tsv')
    src2id_map = joinpath(synsrc_path, 'genome_to_id.tsv')
    err_path = joinpath(bin_path, 'error_analysis')
    src2contig_file = joinpath(err_path, 'src2contig_map.tsv')
    denovo_out_file = glob.glob(joinpath(bin_path, 'clusters.tsv'))[0]
    denovo_errstat_file = joinpath(err_path, 'denovo.errstat.tsv')
    denovo_mean_file = joinpath(err_path, 'denovo.errstat.mean.tsv')

    ##################################################################################################
    # Make working dir
    if not path.exists(err_path):
        makedirs(err_path)

    # Map src contig stats to OTU id
    src_cnt_df = pd.read_csv(src_contig_cnt, sep='\t', header=0)
    src2id_df = pd.read_csv(src2id_map, sep='\t', header=None, names=['CAMI_genomeID', 'file'])
    src_cnt_df['src_id'] = [x.rsplit('/', 1)[1].rsplit('.', 1)[0] for x in src_cnt_df['file']]
    src2id_df['src_id'] = [x.rsplit('/', 1)[1].rsplit('.', 1)[0] for x in src2id_df['file']]
    src_stats_df = src2id_df.merge(src_cnt_df, on='src_id')
    src_stats_df = src_stats_df.rename(columns={'CAMI_genomeID': 'exact_label'})
    # Map genome id and contig id to taxid for error analysis
    sag_taxmap_df = pd.read_csv(sag_tax_map, sep=',', header=0)
    # Map MetaG contigs to their genomes
    mg_contig_map_df = pd.read_csv(mg_contig_map, sep='\t', header=0)
    mg_contig_map_df.columns = ['contig_id', 'exact_label', 'taxid']
    # Merge contig map and taxpath DFs
    tax_mg_df = mg_contig_map_df.merge(sag_taxmap_df, on='exact_label', how='right')
    tax_mg_df = tax_mg_df[['contig_id', 'exact_label', 'species',
                           'strain', 'sample_type']
                          ].query("sample_type == @sample_type")
    # count all bp's for Source genomes, Source MetaG, MockSAGs
    # count all bp's for each read in metaG
    src_metag_cnt_dict = cnt_contig_bp(src_metag_file)
    src_contig_list = list(src_metag_cnt_dict.keys())
    tax_mg_df = tax_mg_df.loc[tax_mg_df['contig_id'].isin(src_contig_list)]
    # Add to tax DF
    tax_mg_df['bp_cnt'] = [src_metag_cnt_dict[x]
                           for x in tax_mg_df['contig_id']
                           ]
    # add src total bp count
    tax_mg_df = tax_mg_df.merge(src_stats_df[['exact_label', 'sum_len']],
                                on='exact_label'
                                ).drop_duplicates()
    tax_mg_df.to_csv(src2contig_file, sep='\t', index=False)

    ###################################################################################################
    # De novo error analysis
    # setup mapping to CAMI ref genomes
    cluster_df = pd.read_csv(denovo_out_file, names=['best_label', 'contig_id'], sep='\t', header=None)
    cluster_trim_df = cluster_df.copy()  # .query('best_label != -1')
    src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
    src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
    src2contig_df['sample_id'] = [x.rsplit('C', 1)[0] for x in src2contig_df['contig_id']]
    contig_bp_df = src2contig_df[['contig_id', 'bp_cnt', 'sample_id']]
    src2contig_df = src2contig_df.drop_duplicates()
    contig_bp_df = contig_bp_df.drop_duplicates()
    clust2src_df = cluster_trim_df.merge(src2contig_df[['contig_id',
                                                        'exact_label',
                                                        'strain',
                                                        'species',
                                                        'bp_cnt']],
                                         on='contig_id', how='left'
                                         )
    clust2src_df['sample_id'] = [x.rsplit('C', 1)[0]
                                 for x in clust2src_df['contig_id']
                                 ]
    grp_clust_df = clust2src_df.groupby(['best_label', 'exact_label', 'strain', 'sample_id']
                                        )['bp_cnt'].sum().reset_index().query("bp_cnt >= 200000")
    grp_clust_list = list(grp_clust_df['best_label'].unique())
    clust2src_df = clust2src_df.query("best_label in @grp_clust_list")
    
    src_bp_dict = {x: y for x, y in zip(src2contig_df['exact_label'],
                                        src2contig_df['sum_len']
                                        )}
    # possible bp's based on asm vs ref genome
    exact2bp_df = src2contig_df[['exact_label', 'strain',
                                 'sample_id', 'sum_len'
                                 ]].copy().drop_duplicates()
    asm2bp_df = src2contig_df.groupby(['exact_label', 'strain', 'sample_id']
                                      )[['bp_cnt']].sum().reset_index()

    poss_bp_df = asm2bp_df.merge(exact2bp_df, on=['exact_label', 'strain', 'sample_id'], how='left')
    poss_bp_df.columns = ['exact_label', 'strain_label', 'sample_id', 'possible_bp', 'total_bp']
    poss_bp_df['asm_per_bp'] = [x / y for x, y in
                                zip(poss_bp_df['possible_bp'],
                                    poss_bp_df['total_bp'])
                                ]
    poss_bp_df['yes_NC'] = [1 if x >= 0.9 else 0 for x in poss_bp_df['asm_per_bp']]
    poss_bp_df['yes_MQ'] = [1 if x >= 0.5 else 0 for x in poss_bp_df['asm_per_bp']]
    poss_bp_df.sort_values(by='asm_per_bp', ascending=False, inplace=True)
    poss_str_bp_df = poss_bp_df[['sample_id', 'strain_label', 'possible_bp',
                                 'total_bp', 'asm_per_bp',
                                 'yes_NC', 'yes_MQ'
                                 ]].copy().drop_duplicates(subset='strain_label')
    # Add taxonomy to each cluster
    clust_tax = []
    for clust in tqdm(clust2src_df['best_label'].unique()):
        samp_id = clust.rsplit('C', 1)[0]
        sub_clust2src_df = clust2src_df.query('sample_id == @samp_id')
        clust_tax.append(cluster2taxonomy([clust, sub_clust2src_df]))
    clust_tax_df = pd.DataFrame(clust_tax, columns=['best_label', 'exact_label', 'strain_label'])
    clust2label_df = clust_tax_df.merge(cluster_trim_df, on='best_label', how='left')
    clust2contig_df = clust2label_df[['best_label', 'contig_id', 'exact_label', 'strain_label'
                                      ]].drop_duplicates()

    # setup multithreading pool
    print("De Novo error analysis started...")
    pool = multiprocessing.Pool(processes=nthreads)
    arg_list = []
    for clust in tqdm(clust2contig_df['best_label'].unique()):
        # subset recruit dataframes
        samp_id = clust.rsplit('C', 1)[0]
        sub_src2cont_df = src2contig_df.query('sample_id == @samp_id')
        sub_contig_bp_df = contig_bp_df.query('sample_id == @samp_id')
        sub_clust_df = clust2contig_df.query('best_label == @clust')
        dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
        # Map Sources/SAGs to Strain IDs
        src_id = sub_clust_df['exact_label'].values[0]
        strain_id = sub_clust_df['strain_label'].values[0]
        src_sub_df = sub_src2cont_df.query('exact_label == @src_id')
        strain_sub_df = sub_src2cont_df.query('strain == @strain_id')
        src2contig_list = list(set(src_sub_df['contig_id'].values))
        src2strain_list = list(set(strain_sub_df['contig_id'].values))
        arg_list.append(['best_label', clust, samp_id, dedup_clust_df, sub_contig_bp_df,
                         src2contig_list, src2strain_list, 'denovo', src_id, strain_id,
                         src_bp_dict
                         ])

    results = pool.imap_unordered(EArecruit, arg_list)
    score_list = []
    for i, output in tqdm(enumerate(results, 1)):
        score_list.extend(output)
    logging.info('\n')
    pool.close()
    pool.join()

    score_df = pd.DataFrame(score_list, columns=['best_label', 'sample_id', 'level', 'algorithm',
                                                 'precision', 'sensitivity', 'MCC', 'F1',
                                                 'N', 'S', 'P', 'TP', 'FP', 'TN', 'FN',
                                                 'possible_bp', 'total_bp'
                                                 ])

    sort_score_df = score_df.sort_values(['best_label', 'sample_id', 'level', 'precision', 'sensitivity'],
                                         ascending=[False, False, False, True, True]
                                         )
    score_tax_df = sort_score_df.merge(clust_tax_df, on='best_label', how='left')
    score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
    score_tax_df['>20Kb'] = 'No'
    score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
    score_tax_df['NC_bins'] = 'No'
    score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                     (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
    score_tax_df['MQ_bins'] = 'No'
    score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                     (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'
    score_tax_df['sample_type'] = sample_type
    score_tax_df.to_csv(denovo_errstat_file, index=False, sep='\t')

    ###########################################################################
    # Compile all results tables from analysis
    completed_files = glob.glob(joinpath(err_path, '*.errstat.tsv'))
    cat_list = []
    for o_file in completed_files:
        err_df = pd.read_csv(o_file, sep='\t', header=0)
        for sample in err_df['sample_id'].unique():
            sub_poss_bp_df = poss_bp_df.query('sample_id == @sample')
            sub_poss_str_bp_df = poss_str_bp_df.query('sample_id == @sample')
            nc_x_poss = sub_poss_bp_df['yes_NC'].sum()
            mq_x_poss = sub_poss_bp_df['yes_MQ'].sum()
            nc_s_poss = sub_poss_str_bp_df['yes_NC'].sum()
            mq_s_poss = sub_poss_str_bp_df['yes_MQ'].sum()
            for algo in err_df['algorithm'].unique():
                for level in err_df['level'].unique():
                    sub_err_df = err_df.query('sample_id == @sample & '
                                              'algorithm == @algo & '
                                              'level == @level')
                    sub_err_df.sort_values(['precision', 'sensitivity'],
                                           ascending=[False, False], inplace=True
                                           )
                    sub_str_df = sub_err_df.drop_duplicates(subset='strain_label')
                    l_20 = '>20Kb'
                    ext_mq_df = sub_err_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                    ext_nc_df = sub_err_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")
                    str_mq_df = sub_str_df.query("NC_bins == 'Yes' | MQ_bins == 'Yes' | @l_20 == 'Yes'")
                    str_nc_df = sub_str_df.query("NC_bins == 'Yes' | @l_20 == 'Yes'")

                    mq_avg_mcc = ext_mq_df['MCC'].mean()
                    nc_avg_mcc = ext_nc_df['MCC'].mean()
                    mq_avg_p = ext_mq_df['precision'].mean()
                    nc_avg_p = ext_nc_df['precision'].mean()
                    mq_avg_r = ext_mq_df['sensitivity'].mean()
                    nc_avg_r = ext_nc_df['sensitivity'].mean()
                    ext_mq_cnt = ext_mq_df['MQ_bins'].count()
                    ext_nc_cnt = ext_nc_df['NC_bins'].count()
                    ext_mq_uniq = len(ext_mq_df['exact_label'].unique())
                    ext_nc_uniq = len(ext_nc_df['exact_label'].unique())
                    str_mq_cnt = str_mq_df['MQ_bins'].count()
                    str_nc_cnt = str_nc_df['NC_bins'].count()
                    str_mq_uniq = len(str_mq_df['strain_label'].unique())
                    str_nc_uniq = len(str_nc_df['strain_label'].unique())

                    err_list = [sample, algo, level, mq_avg_p, mq_avg_r, mq_avg_mcc,
                                nc_avg_p, nc_avg_r, nc_avg_mcc, ext_mq_cnt,
                                ext_mq_uniq, ext_nc_cnt, ext_nc_uniq,
                                str_mq_cnt, str_mq_uniq, str_nc_cnt, str_nc_uniq,
                                mq_x_poss, nc_x_poss, mq_s_poss, nc_s_poss
                                ]
                    cat_list.append(err_list)

    cat_cols = ['sample_id', 'algo', 'level', 'mq_avg_p', 'mq_avg_r', 'mq_avg_mcc',
                'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc', 'ext_mq_cnt',
                'ext_mq_uniq', 'ext_nc_cnt', 'ext_nc_uniq', 'str_mq_cnt',
                'str_mq_uniq', 'str_nc_cnt', 'str_nc_uniq', 'ext_mq_poss',
                'ext_nc_poss', 'str_mq_poss', 'str_nc_poss'
                ]
    # add the total possible NC and MQ bins
    cat_df = pd.DataFrame(cat_list, columns=cat_cols)
    cat_df['sample_type'] = sample_type

    return cat_df
