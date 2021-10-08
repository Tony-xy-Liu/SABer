import difflib
import glob
import logging
import multiprocessing
import sys
from functools import reduce
from os import makedirs, path, listdir
from os.path import isfile
from os.path import join as joinpath

import pandas as pd
from Bio import SeqIO
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def EArecruit(p):  # Error Analysis for all recruits per sag
    col_id, temp_id, temp_clust_df, temp_contig_df, temp_src2contig_list, temp_src2strain_list = p
    temp_clust_df['denovo'] = 1
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
    algo_list = ['denovo']
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
    fa_recs = []
    with open(fasta_file, 'r') as fasta_in:
        for record in SeqIO.parse(fasta_in, 'fasta'):
            f_id = record.id
            # f_description = record.description
            f_seq = str(record.seq)
            if f_seq != '':
                fa_recs.append((f_id, f_seq))

    return fa_recs



##################################################################################################
# INPUT files
saberout_path = sys.argv[1]
synsrc_path = sys.argv[2]
mocksag_path = joinpath(synsrc_path, 'Final_SAGs_20k/')
src_genome_path = joinpath(synsrc_path, 'fasta/')
sag_tax_map = joinpath(synsrc_path, 'genome_taxa_info.tsv')
mg_contig_map = sys.argv[3]  # joinpath(synsrc_path, 'gsa_mapping_pool.binning')
src_metag_file = sys.argv[4]
nthreads = int(sys.argv[5])
err_path = joinpath(saberout_path, 'error_analysis')
sag2cami_file = joinpath(err_path, 'sag2cami_map.tsv')
src2contig_file = joinpath(err_path, 'src2contig_map.tsv')
src2mock_file = joinpath(err_path, 'src_mock_df.tsv')
denovo_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.denovo_clusters.tsv'))[0]
denovo_errstat_file = joinpath(err_path, 'denovo.errstat.tsv')
denovo_mean_file = joinpath(err_path, 'denovo.errstat.mean.tsv')
trusted_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.trusted_clusters.tsv'))[0]
trusted_errstat_file = joinpath(err_path, 'trusted_clusters.errstat.tsv')
trusted_mean_file = joinpath(err_path, 'trusted_clusters.errstat.mean.tsv')
ocsvm_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.ocsvm_clusters.tsv'))[0]
ocsvm_errstat_file = joinpath(err_path, 'ocsvm_clusters.errstat.tsv')
ocsvm_mean_file = joinpath(err_path, 'ocsvm_clusters.errstat.mean.tsv')
inter_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.inter_clusters.tsv'))[0]
inter_errstat_file = joinpath(err_path, 'inter_clusters.errstat.tsv')
inter_mean_file = joinpath(err_path, 'inter_clusters.errstat.mean.tsv')

deit_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.deit_clusters.tsv'))[0]
deit_errstat_file = joinpath(err_path, 'deit.errstat.tsv')
deit_mean_file = joinpath(err_path, 'deit.errstat.mean.tsv')
trit_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.trusted_it.tsv'))[0]
trit_errstat_file = joinpath(err_path, 'trusted_it.errstat.tsv')
trit_mean_file = joinpath(err_path, 'trusted_it.errstat.mean.tsv')
ocit_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.ocsvm_it.tsv'))[0]
ocit_errstat_file = joinpath(err_path, 'ocsvm_it.errstat.tsv')
ocit_mean_file = joinpath(err_path, 'ocsvm_it.errstat.mean.tsv')
init_out_file = glob.glob(joinpath(saberout_path, 'clusters/*.inter_it.tsv'))[0]
init_errstat_file = joinpath(err_path, 'inter_it.errstat.tsv')
init_mean_file = joinpath(err_path, 'inter_it.errstat.mean.tsv')

##################################################################################################
# Make working dir
if not path.exists(err_path):
    makedirs(err_path)

# Map genome id and contig id to taxid for error analysis
sag_taxmap_df = pd.read_csv(sag_tax_map, sep='\t', header=0)
sag_taxmap_df['sp_taxid'] = [int(x) for x in sag_taxmap_df['@@TAXID']]
sag_taxmap_df['sp_name'] = [x.split('|')[-2] for x in sag_taxmap_df['TAXPATHSN']]
taxpath_list = [[str(x) for x in x.split('.')[0].split('|')]
                for x in sag_taxmap_df['TAXPATH']
                ]
taxpath_df = pd.DataFrame(taxpath_list, columns=['domain', 'phylum', 'class', 'order',
                                                 'family', 'genus', 'species', 'strain'
                                                 ])
taxpath_df['CAMI_genomeID'] = [x for x in sag_taxmap_df['_CAMI_genomeID']]
# fix empty species id's
taxpath_df['species'] = [x[1] if str(x[0]) == '' else x[0] for x in
                         zip(taxpath_df['species'], taxpath_df['genus'])
                         ]
# Map MetaG contigs to their genomes
mg_contig_map_df = pd.read_csv(mg_contig_map, sep='\t', header=0)
mg_contig_map_df['TAXID'] = [str(x) for x in mg_contig_map_df['TAXID']]

# Merge contig map and taxpath DFs
tax_mg_df = taxpath_df.merge(mg_contig_map_df, left_on='CAMI_genomeID', right_on='BINID',
                             how='right'
                             )
tax_mg_df = tax_mg_df[['@@SEQUENCEID', 'CAMI_genomeID', 'domain', 'phylum', 'class', 'order',
                       'family', 'genus', 'species', 'strain'
                       ]]

# count all bp's for Source genomes, Source MetaG, MockSAGs
# count all bp's for each read in metaG
src_metag_cnt_dict = cnt_contig_bp(src_metag_file)
# Add to tax DF
tax_mg_df['bp_cnt'] = [src_metag_cnt_dict[x] for x in tax_mg_df['@@SEQUENCEID']]
tax_mg_df.to_csv(src2contig_file, sep='\t', index=False)

if isfile(src2mock_file):
    src_mock_err_df = pd.read_csv(src2mock_file, sep='\t', header=0)
else:
    # list all source genomes
    src_genome_list = [joinpath(src_genome_path, f) for f in listdir(src_genome_path)
                       if ((f.split('.')[-1] == 'fasta' or f.split('.')[-1] == 'fna') and
                           'Sample' not in f)
                       ]
    # list all mockSAGs
    mocksag_list = [joinpath(mocksag_path, f) for f in listdir(mocksag_path)
                    if (f.split('.')[-1] == 'fasta')
                    ]
    src_mock_list = src_genome_list + mocksag_list
    # count total bp's for each src and mock fasta
    fa_bp_cnt_list = []
    for fa_file in mocksag_list:
        f_id = fa_file.split('/')[-1].rsplit('.', 2)[0]
        u_id = fa_file.split('/')[-1].split('.fasta')[0]
        f_type = 'synSAG'
        fa_file, fa_bp_cnt = cnt_total_bp(fa_file)
        print(f_id, u_id, f_type, fa_bp_cnt)
        fa_bp_cnt_list.append([f_id, u_id, f_type, fa_bp_cnt])
    src_bp_cnt_list = []
    for fa_file in src_genome_list:
        f_id = fa_file.split('/')[-1].rsplit('.', 1)[0]
        f_type = 'src_genome'
        fa_file, fa_bp_cnt = cnt_total_bp(fa_file)
        print(f_id, f_type, fa_bp_cnt)
        src_bp_cnt_list.append([f_id, f_type, fa_bp_cnt])
    fa_bp_cnt_df = pd.DataFrame(fa_bp_cnt_list, columns=['sag_id', 'u_id', 'data_type',
                                                         'tot_bp_cnt'
                                                         ])
    src_bp_cnt_df = pd.DataFrame(src_bp_cnt_list, columns=['sag_id', 'data_type',
                                                           'tot_bp_cnt'
                                                           ])
    print(fa_bp_cnt_df.head())
    print(src_bp_cnt_df.head())
    merge_bp_cnt_df = fa_bp_cnt_df.merge(src_bp_cnt_df, on='sag_id', how='left')
    unstack_cnt_df = merge_bp_cnt_df[['u_id', 'tot_bp_cnt_x', 'tot_bp_cnt_y']]
    print(unstack_cnt_df.head())
    unstack_cnt_df.columns = ['sag_id', 'synSAG_tot', 'src_genome_tot']
    # calc basic stats for src and mock
    src_mock_err_list = []
    for ind, row in unstack_cnt_df.iterrows():
        sag_id = row['sag_id']
        mockSAG_tot = row['synSAG_tot']
        src_genome_tot = row['src_genome_tot']
        data_type_list = ['synSAG', 'src_genome']
        for dt in data_type_list:
            algorithm = dt
            for level in ['domain', 'phylum', 'class', 'order',
                          'family', 'genus', 'species', 'strain', 'exact'
                          ]:
                s_m_err_list = [sag_id, algorithm, level, 0, 0, 0, 0]
                if dt == 'synSAG':
                    s_m_err_list[3] += mockSAG_tot  # 'TruePos'
                    s_m_err_list[4] += 0  # 'FalsePos'
                    s_m_err_list[5] += src_genome_tot - mockSAG_tot  # 'FalseNeg'
                    s_m_err_list[6] += 0  # 'TrueNeg'
                    src_mock_err_list.append(s_m_err_list)
    src_mock_err_df = pd.DataFrame(src_mock_err_list, columns=['sag_id', 'algorithm', 'level',
                                                               'TruePos', 'FalsePos',
                                                               'FalseNeg', 'TrueNeg'
                                                               ])
    src_mock_err_df.to_csv(src2mock_file, index=False, sep='\t')

if isfile(sag2cami_file):
    sag2cami_df = pd.read_csv(sag2cami_file, sep='\t', header=0)
else:
    # builds the sag to cami ID mapping file
    mh_list = list(src_mock_err_df['sag_id'].unique())
    cami_list = [str(x) for x in tax_mg_df['CAMI_genomeID'].unique()]
    sag2cami_list = []
    print('Mapping Sources to Synthetic SAGs...')
    for sag_id in mh_list:
        tmp_sag_id = sag_id
        if 'LC_' in tmp_sag_id[:3]:
            tmp_sag_id = tmp_sag_id.strip('LC_')
        match = difflib.get_close_matches(str(tmp_sag_id), cami_list, n=1, cutoff=0)[0]
        m_len = len(match)
        sub_sag_id = tmp_sag_id[:m_len]
        if sub_sag_id != match:
            match = difflib.get_close_matches(str(sub_sag_id), cami_list, n=1, cutoff=0)[0]
            if match == sub_sag_id:
                print("PASSED:", tmp_sag_id, sub_sag_id, match)
            else:
                m1_len = len(match)
                sub_sag_id = tmp_sag_id[:m_len]
                sub_sub_id = sub_sag_id[:m1_len].split('.')[0]
                match = difflib.get_close_matches(str(sub_sub_id), cami_list, n=1, cutoff=0)[0]
                print("PASSED:", tmp_sag_id, sub_sag_id, sub_sub_id, match)
        sag2cami_list.append([sag_id, match])
    sag2cami_df = pd.DataFrame(sag2cami_list, columns=['sag_id', 'CAMI_genomeID'])
    sag2cami_df.to_csv(sag2cami_file, index=False, sep='\t')


###################################################################################################

# setup mapping to CAMI ref genomes
cluster_df = pd.read_csv(denovo_out_file, sep='\t', header=0)
cluster_trim_df = cluster_df.copy()  # .query('best_label != -1')
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
contig_bp_df = src2contig_df[['contig_id', 'bp_cnt']]
clust2src_df = cluster_trim_df.merge(src2contig_df[['contig_id', 'CAMI_genomeID',
                                                    'strain', 'bp_cnt']],
                                     on='contig_id', how='left'
                                     )

# Add taxonomy to each cluster
clust_tax = []
for clust in clust2src_df['best_label'].unique():
    sub_clust_df = clust2src_df.query('best_label == @clust')
    exact_df = sub_clust_df.groupby(['CAMI_genomeID'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID']
    if not ex_label_df.empty:
        exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()
                               ]['CAMI_genomeID'].values[0]
        strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()
                                 ]['strain'].values[0]
        clust_tax.append([clust, exact_label, strain_label])

clust_tax_df = pd.DataFrame(clust_tax, columns=['best_label', 'exact_label', 'strain_label'])
clust2label_df = clust_tax_df.merge(cluster_trim_df, on='best_label', how='left')
clust2contig_df = clust2label_df[['best_label', 'contig_id', 'exact_label', 'strain_label'
                                  ]].drop_duplicates()
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
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
sort_score_df = score_df.sort_values(['best_label', 'level', 'precision', 'sensitivity'],
                                     ascending=[False, False, True, True]
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

stat_mean_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                  'AUC', 'F1']].mean().reset_index()
cnt_bins_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                    'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_exact_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['exact_label']
].nunique().reset_index()
cnt_exact_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_genos_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['strain_label']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'strain_cnt'
                        ]
dfs = [stat_mean_df, cnt_bins_df, cnt_exact_df, cnt_genos_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
score_tax_df.to_csv(denovo_errstat_file, index=False, sep='\t')
stat_df.to_csv(denovo_mean_file, index=False, sep='\t'
               )

##################################################################################################
# Trusted Contigs errstats
# setup mapping to CAMI ref genomes
score_df_list = []
stats_df_list = []

cluster_df = pd.read_csv(trusted_out_file, header=0, sep='\t')

src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(cluster_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = cluster_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
    strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
score_df = score_df.merge(sag2cami_df, left_on='best_label', right_on='sag_id', how='left')
score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                              on='CAMI_genomeID', how='left'
                              )
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

sort_score_df = score_tax_df.sort_values(['best_label', 'level', 'precision',
                                          'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )

stat_mean_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                   'AUC', 'F1']].mean().reset_index()
cnt_bins_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_genos_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['CAMI_genomeID']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_strain_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                       'MQ_bins'])[['strain']
].nunique().reset_index()
cnt_strain_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                         'strain_cnt'
                         ]
dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
sort_score_df.to_csv(trusted_errstat_file, index=False, sep='\t')
stat_df.to_csv(trusted_mean_file, index=False, sep='\t')

##################################################################################################
# OC-SVM Contigs errstats
# setup mapping to CAMI ref genomes
score_df_list = []
stats_df_list = []

cluster_df = pd.read_csv(ocsvm_out_file, header=0, sep='\t')

src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(cluster_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = cluster_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
    strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
score_df = score_df.merge(sag2cami_df, left_on='best_label', right_on='sag_id', how='left')
score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                              on='CAMI_genomeID', how='left'
                              )
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

sort_score_df = score_tax_df.sort_values(['best_label', 'level', 'precision',
                                          'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )

stat_mean_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                   'AUC', 'F1']].mean().reset_index()
cnt_bins_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_genos_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['CAMI_genomeID']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_strain_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                       'MQ_bins'])[['strain']
].nunique().reset_index()
cnt_strain_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                         'strain_cnt'
                         ]
dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
sort_score_df.to_csv(ocsvm_errstat_file, index=False, sep='\t')
stat_df.to_csv(ocsvm_mean_file, index=False, sep='\t')

##################################################################################################
# Combined Contigs errstats
# setup mapping to CAMI ref genomes
score_df_list = []
stats_df_list = []

cluster_df = pd.read_csv(inter_out_file, header=0, sep='\t')

src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(cluster_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = cluster_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
    strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
score_df = score_df.merge(sag2cami_df, left_on='best_label', right_on='sag_id', how='left')
score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                              on='CAMI_genomeID', how='left'
                              )
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

sort_score_df = score_tax_df.sort_values(['best_label', 'level', 'precision',
                                          'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )

stat_mean_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                   'AUC', 'F1']].mean().reset_index()
cnt_bins_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_genos_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['CAMI_genomeID']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_strain_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                       'MQ_bins'])[['strain']
].nunique().reset_index()
cnt_strain_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                         'strain_cnt'
                         ]
dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
sort_score_df.to_csv(inter_errstat_file, index=False, sep='\t')
stat_df.to_csv(inter_mean_file, index=False, sep='\t')

###############################################################################################
# De Novo Iteration 2
# setup mapping to CAMI ref genomes
cluster_df = pd.read_csv(deit_out_file, sep='\t', header=0)
cluster_trim_df = cluster_df.copy()  # .query('best_label != -1')
src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
contig_bp_df = src2contig_df[['contig_id', 'bp_cnt']]
clust2src_df = cluster_trim_df.merge(src2contig_df[['contig_id', 'CAMI_genomeID',
                                                    'strain', 'bp_cnt']],
                                     on='contig_id', how='left'
                                     )
# Add taxonomy to each cluster
clust_tax = []
for clust in tqdm(clust2src_df['best_label'].unique()):
    sub_clust_df = clust2src_df.query('best_label == @clust')
    exact_df = sub_clust_df.groupby(['CAMI_genomeID'])['bp_cnt'].sum().reset_index()
    strain_df = sub_clust_df.groupby(['strain'])['bp_cnt'].sum().reset_index()
    ex_label_df = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()]['CAMI_genomeID']
    if not ex_label_df.empty:
        exact_label = exact_df[exact_df.bp_cnt == exact_df.bp_cnt.max()
                               ]['CAMI_genomeID'].values[0]
        strain_label = strain_df[strain_df.bp_cnt == strain_df.bp_cnt.max()
                                 ]['strain'].values[0]
        clust_tax.append([clust, exact_label, strain_label])

clust_tax_df = pd.DataFrame(clust_tax, columns=['best_label', 'exact_label', 'strain_label'])
clust2label_df = clust_tax_df.merge(cluster_trim_df, on='best_label', how='left')
clust2contig_df = clust2label_df[['best_label', 'contig_id', 'exact_label', 'strain_label'
                                  ]].drop_duplicates()
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
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
sort_score_df = score_df.sort_values(['best_label', 'level', 'precision', 'sensitivity'],
                                     ascending=[False, False, True, True]
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

stat_mean_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                  'AUC', 'F1']].mean().reset_index()
cnt_bins_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                    'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_exact_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['exact_label']
].nunique().reset_index()
cnt_exact_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_genos_df = score_tax_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins'])[['strain_label']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'strain_cnt'
                        ]
dfs = [stat_mean_df, cnt_bins_df, cnt_exact_df, cnt_genos_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
score_tax_df.to_csv(deit_errstat_file, index=False, sep='\t')
stat_df.to_csv(deit_mean_file, index=False, sep='\t')

##################################################################################################
# Trusted Contigs errstats
# setup mapping to CAMI ref genomes
score_df_list = []
stats_df_list = []

cluster_df = pd.read_csv(trit_out_file, header=0, sep='\t')

src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(cluster_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = cluster_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
    strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
score_df = score_df.merge(sag2cami_df, left_on='best_label', right_on='sag_id', how='left')
score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                              on='CAMI_genomeID', how='left'
                              )
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

sort_score_df = score_tax_df.sort_values(['best_label', 'level', 'precision',
                                          'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )

stat_mean_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                   'AUC', 'F1']].mean().reset_index()
cnt_bins_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_genos_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['CAMI_genomeID']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_strain_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                       'MQ_bins'])[['strain']
].nunique().reset_index()
cnt_strain_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                         'strain_cnt'
                         ]
dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
sort_score_df.to_csv(trit_errstat_file, index=False, sep='\t')
stat_df.to_csv(trit_mean_file, index=False, sep='\t')

##################################################################################################
# OC-SVM Contigs errstats
# setup mapping to CAMI ref genomes
score_df_list = []
stats_df_list = []

cluster_df = pd.read_csv(ocit_out_file, header=0, sep='\t')

src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(cluster_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = cluster_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
    strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
score_df = score_df.merge(sag2cami_df, left_on='best_label', right_on='sag_id', how='left')
score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                              on='CAMI_genomeID', how='left'
                              )
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

sort_score_df = score_tax_df.sort_values(['best_label', 'level', 'precision',
                                          'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )

stat_mean_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                   'AUC', 'F1']].mean().reset_index()
cnt_bins_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_genos_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['CAMI_genomeID']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_strain_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                       'MQ_bins'])[['strain']
].nunique().reset_index()
cnt_strain_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                         'strain_cnt'
                         ]
dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
sort_score_df.to_csv(ocit_errstat_file, index=False, sep='\t')
stat_df.to_csv(ocit_mean_file, index=False, sep='\t')

##################################################################################################
# Combined Contigs errstats
# setup mapping to CAMI ref genomes
score_df_list = []
stats_df_list = []

cluster_df = pd.read_csv(init_out_file, header=0, sep='\t')

src2contig_df = pd.read_csv(src2contig_file, header=0, sep='\t')
src2contig_df = src2contig_df.rename(columns={'@@SEQUENCEID': 'contig_id'})
sag2cami_df = pd.read_csv(sag2cami_file, header=0, sep='\t')
sag2contig_df = sag2cami_df.merge(src2contig_df, on='CAMI_genomeID', how='left')

# setup multithreading pool
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for clust in tqdm(cluster_df['best_label'].unique()):
    # subset recruit dataframes
    sub_clust_df = cluster_df.query('best_label == @clust')
    dedup_clust_df = sub_clust_df[['best_label', 'contig_id']].drop_duplicates()
    # Map Sources/SAGs to Strain IDs
    src_id = sag2contig_df.query('sag_id == @clust')['CAMI_genomeID'].values[0]
    strain_id = sag2contig_df.query('sag_id == @clust')['strain'].values[0]
    src_sub_df = src2contig_df.query('CAMI_genomeID == @src_id')
    strain_sub_df = src2contig_df.query('strain == @strain_id')
    src2contig_list = list(set(src_sub_df['contig_id'].values))
    src2strain_list = list(set(strain_sub_df['contig_id'].values))
    arg_list.append(['best_label', clust, dedup_clust_df, contig_bp_df, src2contig_list,
                     src2strain_list
                     ])

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
score_df = score_df.merge(sag2cami_df, left_on='best_label', right_on='sag_id', how='left')
score_tax_df = score_df.merge(clust2src_df[['CAMI_genomeID', 'strain']].drop_duplicates(),
                              on='CAMI_genomeID', how='left'
                              )
score_tax_df['size_bp'] = score_tax_df['TP'] + score_tax_df['FP']
score_tax_df['>20Kb'] = 'No'
score_tax_df.loc[score_tax_df['size_bp'] >= 20000, '>20Kb'] = 'Yes'
score_tax_df['NC_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.95) &
                 (score_tax_df['sensitivity'] >= 0.9), 'NC_bins'] = 'Yes'
score_tax_df['MQ_bins'] = 'No'
score_tax_df.loc[(score_tax_df['precision'] >= 0.9) &
                 (score_tax_df['sensitivity'] >= 0.5), 'MQ_bins'] = 'Yes'

sort_score_df = score_tax_df.sort_values(['best_label', 'level', 'precision',
                                          'sensitivity'],
                                         ascending=[False, False, True, True]
                                         )

stat_mean_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['precision', 'sensitivity', 'MCC',
                                                   'AUC', 'F1']].mean().reset_index()
cnt_bins_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                     'MQ_bins']).size().reset_index()
cnt_bins_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                       'bin_cnt'
                       ]
cnt_genos_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                      'MQ_bins'])[['CAMI_genomeID']
].nunique().reset_index()
cnt_genos_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                        'genome_cnt'
                        ]
cnt_strain_df = sort_score_df.groupby(['level', 'algorithm', '>20Kb', 'NC_bins',
                                       'MQ_bins'])[['strain']
].nunique().reset_index()
cnt_strain_df.columns = ['level', 'algorithm', '>20Kb', 'NC_bins', 'MQ_bins',
                         'strain_cnt'
                         ]
dfs = [stat_mean_df, cnt_bins_df, cnt_genos_df, cnt_strain_df]
stat_df = reduce(lambda left, right: pd.merge(left, right, on=['level',
                                                               'algorithm', '>20Kb',
                                                               'NC_bins', 'MQ_bins'
                                                               ]), dfs
                 )
sort_score_df.to_csv(init_errstat_file, index=False, sep='\t')
stat_df.to_csv(init_mean_file, index=False, sep='\t')
