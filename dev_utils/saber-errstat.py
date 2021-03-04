import sys
from os import makedirs, path, listdir
from os.path import isfile
from os.path import join as joinpath

import numpy as np
import pandas as pd
from Bio import SeqIO

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
from tqdm import tqdm
import multiprocessing
import glob


def calc_err(df):
    # build error type df for each filter separately
    group_df = df.copy()
    group_df['precision'] = group_df['TruePos'] / \
                            (group_df['TruePos'] + group_df['FalsePos'])
    group_df['sensitivity'] = group_df['TruePos'] / \
                              (group_df['TruePos'] + group_df['FalseNeg'])
    group_df['specificity'] = group_df['TrueNeg'] / \
                              (group_df['TrueNeg'] + group_df['FalsePos'])
    group_df['type1_error'] = group_df['FalsePos'] / \
                              (group_df['FalsePos'] + group_df['TrueNeg'])
    group_df['type2_error'] = group_df['FalseNeg'] / \
                              (group_df['FalseNeg'] + group_df['TruePos'])
    group_df['F1_score'] = 2 * ((group_df['precision'] * group_df['sensitivity']) / \
                                (group_df['precision'] + group_df['sensitivity']))

    # Calc MCC, two ways depending on the denom
    group_df['sum1'] = group_df['TruePos'] + group_df['FalsePos']
    group_df['sum2'] = group_df['TruePos'] + group_df['FalseNeg']
    group_df['sum3'] = group_df['TrueNeg'] + group_df['FalsePos']
    group_df['sum4'] = group_df['TrueNeg'] + group_df['FalseNeg']
    group_df['denom'] = [(x[0] * x[1] * x[2] * x[3]) ** (1 / 2)
                         if (x[0] != 0 & x[1] != 0 & x[2] != 0 & x[3] != 0)
                         else 1 for x in zip(group_df['sum1'], group_df['sum2'],
                                             group_df['sum3'], group_df['sum4']
                                             )]
    group_df['MCC'] = ((group_df['TruePos'] * group_df['TrueNeg']) - \
                       (group_df['FalsePos'] * group_df['FalseNeg'])) / group_df['denom']

    # group_df['N'] = group_df['TrueNeg'] + group_df['TruePos'] + \
    #                group_df['FalseNeg'] + group_df['FalsePos']
    # group_df['S'] = (group_df['TruePos'] + group_df['FalseNeg']) / group_df['N']
    # group_df['P'] = (group_df['TruePos'] + group_df['FalsePos']) / group_df['N']
    # group_df['MCC'] = ((group_df['TruePos'] / group_df['N']) - group_df['S'] * group_df['P']) / \
    #                  ((group_df['S'] * group_df['P']) * (1 - group_df['S']) * (1 - group_df['P'])) ** (1 / 2)
    group_df.set_index(['sag_id', 'algorithm', 'level'], inplace=True)
    stats_df = group_df[['precision', 'sensitivity', 'specificity', 'type1_error',
                         'type2_error', 'F1_score', 'MCC']]
    stack_df = stats_df.stack().reset_index()
    stack_df.columns = ['sag_id', 'algorithm', 'level', 'statistic', 'score']
    return stack_df


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


def cnt_total_bp(fasta_file):
    # counts total basepairs contained in file
    # returns fasta_file name and total counts for entire fasta file

    fasta_records = get_seqs(fasta_file)
    bp_sum = 0
    for f_rec in fasta_records:
        bp_sum += len(f_rec[1])
    return fasta_file, bp_sum


def cnt_contig_bp(fasta_file):
    # counts basepairs/read contained in file
    # returns dictionary of {read_header:bp_count}

    fasta_records = get_seqs(fasta_file)
    fa_cnt_dict = {}
    for f_rec in fasta_records:
        fa_cnt_dict[f_rec[0]] = len(f_rec[1])

    return fa_cnt_dict


def collect_error(p):
    error_list = []
    tp_list = []
    sag_id, tax_mg_df, sag_sub_df, algo_list, level_list = p
    sag_key_list = [str(s) for s in set(tax_mg_df['CAMI_genomeID']) if str(s) in sag_id]
    sag_key = max(sag_key_list, key=len)
    for algo in algo_list:
        algo_sub_df = sag_sub_df.loc[sag_sub_df[algo] == 1]
        algo_include_contigs = pd.DataFrame(algo_sub_df['contig_id'], columns=['contig_id'])
        for col in level_list:
            col_key = tax_mg_df.loc[tax_mg_df['CAMI_genomeID'] == sag_key, col].iloc[0]
            cami_include_ids = pd.DataFrame(
                tax_mg_df.loc[tax_mg_df[col] == col_key]['CAMI_genomeID'].unique(),
                columns=['CAMI_genomeID']
            )
            # mg_include_contigs = sag_sub_df.loc[sag_sub_df['CAMI_genomeID'].isin(
            #                                    cami_include_ids)]['@@SEQUENCEID']
            mg_include_contigs = tax_mg_df.merge(cami_include_ids, how='inner',
                                                 on='CAMI_genomeID'
                                                 )['@@SEQUENCEID']
            sag_key_df = pd.DataFrame([sag_key], columns=['CAMI_genomeID'])
            # sag_include_contigs = sag_sub_df.loc[sag_sub_df['CAMI_genomeID'].isin(
            #                                    cami_include_ids)]['@@SEQUENCEID']
            sag_include_contigs = tax_mg_df.merge(sag_key_df, how='inner',
                                                  on='CAMI_genomeID')['@@SEQUENCEID']
            if col == 'CAMI_genomeID':
                col = 'exact'
                col_key = sag_key
            err_list = [sag_id, algo, col, 0, 0, 0, 0]
            # for contig_id, contig_count in zip(tax_mg_df['@@SEQUENCEID'], tax_mg_df['bp_cnt']):
            # Pos_cnt_df = tax_mg_df.loc[tax_mg_df['@@SEQUENCEID'].isin(algo_include_contigs)]
            Pos_cnt_df = tax_mg_df.merge(algo_include_contigs, how='inner', right_on='contig_id',
                                         left_on='@@SEQUENCEID'
                                         )
            Neg_cnt_df = tax_mg_df.loc[~tax_mg_df['@@SEQUENCEID'].isin(Pos_cnt_df['@@SEQUENCEID'])]

            # TP_cnt_df = Pos_cnt_df.loc[Pos_cnt_df['@@SEQUENCEID'].isin(mg_include_contigs)]
            TP_cnt_df = Pos_cnt_df.merge(mg_include_contigs, how='inner', right_on='@@SEQUENCEID',
                                         left_on='@@SEQUENCEID'
                                         )
            FP_cnt_df = Pos_cnt_df.loc[~Pos_cnt_df['@@SEQUENCEID'].isin(mg_include_contigs)]

            # FN_cnt_df = Neg_cnt_df.loc[Neg_cnt_df['@@SEQUENCEID'].isin(sag_include_contigs)]
            FN_cnt_df = Neg_cnt_df.merge(sag_include_contigs, how='inner', right_on='@@SEQUENCEID',
                                         left_on='@@SEQUENCEID'
                                         )
            TN_cnt_df = Neg_cnt_df.loc[~Neg_cnt_df['@@SEQUENCEID'].isin(sag_include_contigs)]
            err_list[3] = TP_cnt_df['bp_cnt'].sum()  # 'TruePos'
            err_list[4] = FP_cnt_df['bp_cnt'].sum()  # 'FalsePos'
            err_list[5] = FN_cnt_df['bp_cnt'].sum()  # 'FalseNeg'
            err_list[6] = TN_cnt_df['bp_cnt'].sum()  # 'TrueNeg'
            error_list.append(err_list)
            if col == 'strain':
                tp_df = TP_cnt_df.copy()
                tp_df['sag_id'] = sag_id
                tp_df['algo'] = algo
                tp_list.append(tp_df)

    return error_list, tp_list


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
##################################################################################################


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

err_path = saberout_path + '/error_analysis'
if not path.exists(err_path):
    makedirs(err_path)

tax_mg_df.to_csv(saberout_path + 'error_analysis/src2sag_map.tsv', sep='\t', index=False)

# count all bp's for Source genomes, Source MetaG, MockSAGs
# count all bp's for each read in metaG
src_metag_cnt_dict = cnt_contig_bp(src_metag_file)
# Add to tax DF
tax_mg_df['bp_cnt'] = [src_metag_cnt_dict[x] for x in tax_mg_df['@@SEQUENCEID']]

if isfile(saberout_path + 'error_analysis/src_mock_df.tsv'):
    src_mock_err_df = pd.read_csv(saberout_path + 'error_analysis/src_mock_df.tsv',
                                  sep='\t', header=0
                                  )
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
    # unstack_cnt_df = fa_bp_cnt_df.set_index(['sag_id', 'data_type']).unstack(level=-1).reset_index()
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

    src_mock_err_df.to_csv(saberout_path + 'error_analysis/src_mock_df.tsv', index=False, sep='\t')

# MinHash
mh_file = glob.glob(joinpath(saberout_path, 'minhash_recruits/*.mhr_trimmed_recruits.tsv'))[0]
mh_concat_df = pd.read_csv(mh_file, sep='\t', header=0)
mh_concat_df = mh_concat_df[['sag_id', 'contig_id']]

# MBN Abundance
mbn_file = glob.glob(joinpath(saberout_path, 'abund_recruits/*.abr_trimmed_recruits.tsv'))[0]
mbn_concat_df = pd.read_csv(mbn_file, sep='\t', header=0)
mbn_concat_df = mbn_concat_df[['sag_id', 'contig_id']]

# Tetra GMM
gmm_file = glob.glob(joinpath(saberout_path, 'tetra_recruits/*.gmm.tra_trimmed_recruits.tsv'))[0]
gmm_concat_df = pd.read_csv(gmm_file, sep='\t', header=0)
gmm_concat_df = gmm_concat_df[['sag_id', 'contig_id']]

# Tetra OCSVM
svm_file = glob.glob(joinpath(saberout_path, 'tetra_recruits/*.svm.tra_trimmed_recruits.tsv'))[0]
svm_concat_df = pd.read_csv(svm_file, sep='\t', header=0)
svm_concat_df = svm_concat_df[['sag_id', 'contig_id']]

# Tetra Isolation Forest
iso_file = glob.glob(joinpath(saberout_path, 'tetra_recruits/*.iso.tra_trimmed_recruits.tsv'))[0]
iso_concat_df = pd.read_csv(iso_file, sep='\t', header=0)
iso_concat_df = iso_concat_df[['sag_id', 'contig_id']]

# Tetra Combined
comb_file = glob.glob(joinpath(saberout_path, 'tetra_recruits/*.comb.tra_trimmed_recruits.tsv'))[0]
comb_concat_df = pd.read_csv(comb_file, sep='\t', header=0)
comb_concat_df = comb_concat_df[['sag_id', 'contig_id']]

# Combined xPG Recruits
xpg_file = joinpath(saberout_path, 'xPGs/CONTIG_MAP.xPG.tsv')
print('loading Combined combined files')
xpg_df = pd.read_csv(xpg_file, sep='\t', header=0,  # index_col=0,
                     names=['sag_id', 'contig_id']
                     )
xpg_df = xpg_df[['sag_id', 'contig_id']]

mh_concat_df['algorithm'] = 'minhash'
mbn_concat_df['algorithm'] = 'mbn_abund'
gmm_concat_df['algorithm'] = 'tetra_gmm'
svm_concat_df['algorithm'] = 'tetra_svm'
iso_concat_df['algorithm'] = 'tetra_iso'
comb_concat_df['algorithm'] = 'tetra_comb'
xpg_df['algorithm'] = 'xpg'

mh_concat_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)
mbn_concat_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)
gmm_concat_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)
svm_concat_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)
iso_concat_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)
comb_concat_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)
xpg_df.drop_duplicates(subset=['sag_id', 'contig_id'], inplace=True)

sag_set = set(mh_concat_df['sag_id']).intersection(set(xpg_df['sag_id']), set(mbn_concat_df['sag_id']),
                                                   set(gmm_concat_df['sag_id']), set(svm_concat_df['sag_id']),
                                                   set(iso_concat_df['sag_id']), set(comb_concat_df['sag_id'])
                                                   )
final_concat_df = pd.concat([mh_concat_df, mbn_concat_df,
                             gmm_concat_df, svm_concat_df,
                             iso_concat_df,
                             comb_concat_df,
                             xpg_df
                             ])
#final_concat_df = final_concat_df.loc[final_concat_df['sag_id'].isin(list(sag_set))]
final_concat_df['predict'] = 1
piv_table_df = pd.pivot_table(final_concat_df, columns=['algorithm'], index=['sag_id', 'contig_id'],
                              values=['predict'], aggfunc=np.sum, fill_value=0
                              )
piv_table_df.reset_index(inplace=True)
piv_table_df.columns = [''.join(col).replace('predict', '') for col in
                        piv_table_df.columns.values
                        ]

algo_list = ['minhash', 'mbn_abund', 'tetra_gmm', 'tetra_svm', 'tetra_iso', 'tetra_comb', 'xpg']
level_list = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain',
              'CAMI_genomeID'
              ]
####
pool = multiprocessing.Pool(processes=nthreads)
arg_list = []
for sag_id in tqdm(piv_table_df['sag_id'].unique()):
    sag_sub_df = piv_table_df.loc[piv_table_df['sag_id'] == sag_id]
    arg_list.append([sag_id, tax_mg_df, sag_sub_df, algo_list, level_list])
results = pool.imap_unordered(collect_error, arg_list)

tot_error_list = []
tot_tp_list = []
for output in tqdm(results):
    tot_error_list.extend(output[0])
    tot_tp_list.extend(output[1])

pool.close()
pool.join()
####

mbn_concat_df = pd.concat(tot_tp_list)
mbn_concat_df.to_csv(err_path + '/TruePos_table.tsv', index=False, sep='\t')

mg_err_df = pd.DataFrame(tot_error_list, columns=['sag_id', 'algorithm', 'level',
                                                  'TruePos', 'FalsePos',
                                                  'FalseNeg', 'TrueNeg'
                                                  ])

final_err_df = pd.concat([src_mock_err_df, mg_err_df])
final_err_df.to_csv(err_path + '/All_error_count.tsv', index=False, sep='\t')

calc_stats_df = calc_err(final_err_df)
stat_list = ['precision', 'sensitivity', 'F1_score', 'MCC']
calc_stats_df = calc_stats_df.loc[calc_stats_df['statistic'].isin(stat_list)]
calc_stats_df.to_csv(err_path + '/All_stats_count.tsv', index=False, sep='\t')
