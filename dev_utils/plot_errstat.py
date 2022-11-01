import os
import sys
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sci_stats
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# specify that all columns should be shown
pd.set_option('max_columns', None)

# plot aestetics
sns.set_context("paper")


def calc_f1score(P, R):
    F1_score = 2 * ((P * R) / (P + R))

    return F1_score


saber_single_file = sys.argv[1]
saber_multi_file = sys.argv[2]
unitem_single_file = sys.argv[3]
unitem_multi_file = sys.argv[4]
vamb_multi_file = sys.argv[5]
diffdna_single_file = sys.argv[6]
diffdna_multi_file = sys.argv[7]
# working directory
workdir = sys.argv[8]

# column renaming/mapping dictionaries
type2label = {'Airways': 'Air',
              'Gastrointestinal': 'GI',
              'Oral': 'Oral',
              'Skin': 'Skin',
              'Urogenital': 'Urog',
              }
algo2rank = {'denovo': 0, 'hdbscan': 1,
             'ocsvm': 2, 'intersect': 3,
             'xPG': 4
             }
type2rank = {'Air': 0,
             'GI': 1,
             'Oral': 2,
             'Skin': 3,
             'Urog': 4,
             }
mode2rank = {'algo_defaults': 0,
             'majority_rule': 1,
             'best_cluster': 2,
             'best_match': 3             
             }
param2rank = {'algo_defaults': 0,
              'very_relaxed': 1,
              'relaxed': 2,
              'strict': 3,
              'very_strict': 4
              }
binnerconf2rank = {'maxbin_ms40': 0,
                   'maxbin_ms107': 1,
                   'metabat_specific': 2,
                   'metabat_veryspecific': 3,
                   'metabat_superspecific': 4,
                   'metabat_sensitive': 5,
                   'metabat_verysensitive': 6,
                   'metabat2': 7
                   }
binner2rank = {'maxbin': 0,
               'metabat': 1,
               'metabat2': 2,
               'VAMB': 3,
               'SABer_denovo': 4,
               'SABer_intersect': 5,
               'SABer_xPG': 6,
               'SABer_hdbscan': 7,
               'SABer_ocsvm': 8
               }

level2rank = {'exact_assembly_single': 0,
              'exact_absolute_single': 1,
              'strain_assembly_single': 2,
              'strain_absolute_single': 3,
              'exact_assembly_multi': 4,
              'exact_absolute_multi': 5,
              'strain_assembly_multi': 6,
              'strain_absolute_multi': 7
              }
cmap = sns.color_palette()
cmap_muted = sns.color_palette("muted")
cmap_pastel = sns.color_palette("pastel")
binner2cmap = {'maxbin': cmap_pastel[0],
               'metabat': cmap_pastel[9],
               'metabat2': cmap_pastel[2],
               'VAMB': cmap_pastel[4],
               'SABer_denovo': cmap_pastel[3],
               'SABer_intersect': cmap_pastel[1],
               'SABer_xPG': cmap[1],
                'SABer_hdbscan': cmap_pastel[1],
               'SABer_ocsvm': cmap_pastel[1]
               }

# Load stats tables
saber_single_df = pd.read_csv(saber_single_file, header=0, sep='\t')
saber_multi_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
unitem_single_df = pd.read_csv(unitem_single_file, header=0, sep='\t')
unitem_multi_df = pd.read_csv(unitem_multi_file, header=0, sep='\t')
vamb_multi_df = pd.read_csv(vamb_multi_file, header=0, sep='\t')
diffdna_single_df = pd.read_csv(diffdna_single_file, sep='\t',
                                header=0
                                )
diffdna_multi_df = pd.read_csv(diffdna_multi_file, sep='\t',
                               header=0
                               )
# Unify table formats
# col_order = ['binner', 'bin_mode', 'level', 'sample_type', 'sample_id',
#             'best_label', 'precision', 'sensitivity', 'MCC', 'F1',
#             'possible_bp', 'total_bp', 'exact_label', 'strain_label',
#             '>20Kb', 'NC_bins', 'MQ_bins'
#             ]
# SABer first
diffdna_single_df['sample_id'] = ['S' + str(x) for x in
                                  diffdna_single_df['sample_id']
                                  ]
diffdna_multi_df['sample_id'] = ['S' + str(x) for x in
                                 diffdna_multi_df['sample_id']
                                 ]
diffdna_single_df['ref_id'] = [x.rsplit('.', 1)[0] for x in
                                diffdna_single_df['ref_id']
                                ]
diffdna_multi_df['ref_id'] = [x.rsplit('.', 1)[0] for x in
                                diffdna_multi_df['ref_id']
                                ]

saber_single_df['binner'] = ['_'.join(['SABer', str(x), str(y), str(z)])
                             for x, y, z in
                             zip(saber_single_df['algorithm'],
                                 saber_single_df['mode'],
                                 saber_single_df['param_set']
                                 )
                             ]
saber_single_df['bin_mode'] = 'single'
saber_single_df['sample_id'] = ['S' + str(x) for x in
                                saber_single_df['sample_id']
                                ]
saber_s_df = saber_single_df  # .drop(columns=['algorithm', 'mode', 'param_set']
#      )[col_order]
saber_multi_df['binner'] = ['_'.join(['SABer', str(x), str(y), str(z)])
                            for x, y, z in
                            zip(saber_multi_df['algorithm'],
                                saber_multi_df['mode'],
                                saber_multi_df['param_set']
                                )
                            ]
saber_multi_df['bin_mode'] = 'multi'
saber_multi_df['sample_id'] = ['S' + str(x) for x in
                               saber_multi_df['sample_id']
                               ]
saber_m_df = saber_multi_df  # .drop(columns=['algorithm', 'mode', 'param_set']
#      )[col_order]

# UniteM Binners
unitem_single_df['bin_mode'] = 'single'
unitem_single_df['sample_id'] = ['S' + str(x) for x in
                                 unitem_single_df['sample_id']
                                 ]
unitem_s_df = unitem_single_df  # .drop(columns=['algorithm'])[col_order]
unitem_multi_df['bin_mode'] = 'multi'
unitem_multi_df['sample_id'] = ['S' + str(x) for x in
                                unitem_multi_df['sample_id']
                                ]
unitem_m_df = unitem_multi_df  # .drop(columns=['algorithm'])[col_order]

# VAMB
vamb_multi_df['bin_mode'] = 'multi'
vamb_multi_df['binner'] = 'VAMB'
vamb_multi_df['sample_id'] = ['S' + str(x) for x in
                              vamb_multi_df['sample_id']
                              ]
vamb_m_df = vamb_multi_df  #.drop(columns=['algorithm'])[col_order]

bin_cat_df = pd.concat([saber_s_df, saber_m_df,
                        unitem_s_df, unitem_m_df,
                        vamb_m_df
                        ])
bin_cat_df['binner_config'] = [x + '_' + y for x, y in zip(bin_cat_df['binner'],
                                                           bin_cat_df['bin_mode']
                                                           )]
bin_cat_df['level_mode'] = [x + '_' + y for x, y in zip(bin_cat_df['level'],
                                                        bin_cat_df['bin_mode']
                                                        )]
bin_cat_df['dataset'] = [type2label[x] for x in bin_cat_df['sample_type']]
########################################################################################################################
##### Calc all basic metrics ###########################################################################################
########################################################################################################################
'''
# By dataset
cat_list = []
for binner in bin_cat_df['binner'].unique():
    binner_df = bin_cat_df.query('binner == @binner')
    for bin_mode in binner_df['bin_mode'].unique():
        bin_mode_df = binner_df.query('bin_mode == @bin_mode')
        for dataset in bin_mode_df['dataset'].unique():
            dataset_df = bin_mode_df.query('dataset == @dataset')
            for level in dataset_df['level'].unique():
                sub_err_df = dataset_df.query('level == @level')
                if sub_err_df.shape[0] != 0:
                    print(binner, bin_mode, dataset, level, sub_err_df.shape)
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
                    sub_err_df['asm_per_bp'] = [x / y for x, y in
                                                zip(sub_err_df['possible_bp'],
                                                    sub_err_df['total_bp'])
                                                ]
                    sub_err_df['yes_NC'] = [1 if x >= 0.9 else 0 for x in sub_err_df['asm_per_bp']]
                    sub_err_df['yes_MQ'] = [1 if x >= 0.5 else 0 for x in sub_err_df['asm_per_bp']]
                    sub_err_df.sort_values(by='asm_per_bp', ascending=False, inplace=True)
                    poss_ext_bp_df = sub_err_df[['exact_label', 'possible_bp',
                                                 'total_bp', 'asm_per_bp',
                                                 'yes_NC', 'yes_MQ'
                                                 ]].copy().drop_duplicates(subset='exact_label')
                    poss_str_bp_df = sub_err_df[['strain_label', 'possible_bp',
                                                 'total_bp', 'asm_per_bp',
                                                 'yes_NC', 'yes_MQ'
                                                 ]].copy().drop_duplicates(subset='strain_label')
                    ext_mq_poss = poss_ext_bp_df['yes_MQ'].sum()
                    ext_nc_poss = poss_ext_bp_df['yes_NC'].sum()
                    str_mq_poss = poss_str_bp_df['yes_MQ'].sum()
                    str_nc_poss = poss_str_bp_df['yes_NC'].sum()
                    err_list = [binner, bin_mode, level, dataset, mq_avg_p,
                                mq_avg_r, mq_avg_mcc, nc_avg_p, nc_avg_r,
                                nc_avg_mcc, ext_mq_cnt, ext_mq_uniq,
                                ext_nc_cnt, ext_nc_uniq, str_mq_cnt, str_mq_uniq,
                                str_nc_cnt, str_nc_uniq, ext_mq_poss, ext_nc_poss,
                                str_mq_poss, str_nc_poss
                                ]
                    cat_list.append(err_list)
cat_cols = ['binner', 'bin_mode', 'level', 'dataset', 'mq_avg_p', 'mq_avg_r',
            'mq_avg_mcc', 'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc', 'ext_mq_cnt',
            'ext_mq_uniq', 'ext_nc_cnt', 'ext_nc_uniq', 'str_mq_cnt',
            'str_mq_uniq', 'str_nc_cnt', 'str_nc_uniq', 'ext_mq_poss',
            'ext_nc_poss', 'str_mq_poss', 'str_nc_poss'
            ]
dataset_metrics_df = pd.DataFrame(cat_list, columns=cat_cols)
dataset_metrics_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.dataset.avg_metrics.tsv'), sep='\t', index=False)

# By sample
cat_list = []
for binner in bin_cat_df['binner'].unique():
    binner_df = bin_cat_df.query('binner == @binner')
    for bin_mode in binner_df['bin_mode'].unique():
        bin_mode_df = binner_df.query('bin_mode == @bin_mode')
        for dataset in bin_mode_df['dataset'].unique():
            dataset_df = bin_mode_df.query('dataset == @dataset')
            for level in dataset_df['level'].unique():
                level_df = dataset_df.query('level == @level')
                for sample_id in level_df['sample_id'].unique():
                    sub_err_df = level_df.query('binner == @binner & bin_mode == @bin_mode & '
                                                'dataset == @dataset & level == @level & '
                                                'sample_id == @sample_id'
                                                )
                    if sub_err_df.shape[0] != 0:
                        print(binner, bin_mode, dataset, level, sample_id, sub_err_df.shape)
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
                        sub_err_df['asm_per_bp'] = [x / y for x, y in
                                                    zip(sub_err_df['possible_bp'],
                                                        sub_err_df['total_bp'])
                                                    ]
                        sub_err_df['yes_NC'] = [1 if x >= 0.9 else 0 for x in sub_err_df['asm_per_bp']]
                        sub_err_df['yes_MQ'] = [1 if x >= 0.5 else 0 for x in sub_err_df['asm_per_bp']]
                        sub_err_df.sort_values(by='asm_per_bp', ascending=False, inplace=True)
                        poss_ext_bp_df = sub_err_df[['exact_label', 'possible_bp',
                                                 'total_bp', 'asm_per_bp',
                                                 'yes_NC', 'yes_MQ'
                                                 ]].copy().drop_duplicates(subset='exact_label')
                        poss_str_bp_df = sub_err_df[['strain_label', 'possible_bp',
                                                     'total_bp', 'asm_per_bp',
                                                     'yes_NC', 'yes_MQ'
                                                     ]].copy().drop_duplicates(subset='strain_label')
                        ext_mq_poss = poss_ext_bp_df['yes_MQ'].sum()
                        ext_nc_poss = poss_ext_bp_df['yes_NC'].sum()
                        str_mq_poss = poss_str_bp_df['yes_MQ'].sum()
                        str_nc_poss = poss_str_bp_df['yes_NC'].sum()
                        err_list = [binner, bin_mode, level, dataset, sample_id, mq_avg_p,
                                    mq_avg_r, mq_avg_mcc, nc_avg_p, nc_avg_r,
                                    nc_avg_mcc, ext_mq_cnt, ext_mq_uniq,
                                    ext_nc_cnt, ext_nc_uniq, str_mq_cnt, str_mq_uniq,
                                    str_nc_cnt, str_nc_uniq, ext_mq_poss, ext_nc_poss,
                                    str_mq_poss, str_nc_poss
                                    ]
                        cat_list.append(err_list)
cat_cols = ['binner', 'bin_mode', 'level', 'dataset', 'sample_id', 'mq_avg_p', 'mq_avg_r',
            'mq_avg_mcc', 'nc_avg_p', 'nc_avg_r', 'nc_avg_mcc', 'ext_mq_cnt',
            'ext_mq_uniq', 'ext_nc_cnt', 'ext_nc_uniq', 'str_mq_cnt',
            'str_mq_uniq', 'str_nc_cnt', 'str_nc_uniq', 'ext_mq_poss',
            'ext_nc_poss', 'str_mq_poss', 'str_nc_poss'
            ]
sample_metrics_df = pd.DataFrame(cat_list, columns=cat_cols)
sample_metrics_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.sample.avg_metrics.tsv'), sep='\t', index=False)
'''
dataset_metrics_df = pd.read_csv(os.path.join(workdir, 'tables/ALL_BINNERS.dataset.avg_metrics.tsv'), sep='\t',
                                 header=0)

sample_metrics_df = pd.read_csv(os.path.join(workdir, 'tables/ALL_BINNERS.sample.avg_metrics.tsv'), sep='\t',
                                header=0)

# below should be kept in the above processing in the future
dataset_metrics_df['level_mode'] = [x + '_' + y for x, y in zip(dataset_metrics_df['level'],
                                                                dataset_metrics_df['bin_mode']
                                                                )]
sample_metrics_df['level_mode'] = [x + '_' + y for x, y in zip(sample_metrics_df['level'],
                                                               sample_metrics_df['bin_mode']
                                                               )]
dataset_metrics_df['binner_config'] = [x + '_' + y for x, y in zip(dataset_metrics_df['binner'],
                                                                   dataset_metrics_df['bin_mode']
                                                                   )]
sample_metrics_df['binner_config'] = [x + '_' + y for x, y in zip(sample_metrics_df['binner'],
                                                                  sample_metrics_df['bin_mode']
                                                                  )]

########################################################################################################################
##### RUN NC STATS #####################################################################################################
########################################################################################################################
print('############################################################')
print("RUN NC STATS")
print('############################################################')
cnt_df_list = []
for level_mode in sample_metrics_df['level_mode'].unique():
    print('############################################################')
    print(f"The Level tested is {level_mode}")
    print('############################################################')
    sub_df = sample_metrics_df.query("level_mode == @level_mode")
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = sci_stats.f_oneway(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_nc_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    m_comp = pairwise_tukeyhsd(endog=sub_df['ext_nc_uniq'],
                               groups=sub_df['binner_config'],
                               alpha=0.05
                               )
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    stat, p = sci_stats.kruskal(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_nc_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    '''
    # Wilcoxon signed rank test
    bc_list = sub_df['binner_config'].unique()
    for bc0 in bc_list:
        bc0_df = sub_df.query('binner_config == @bc0')
        for bc1 in bc_list:
            if ((bc0 != bc1) &
                    (bc0.split('_', 1)[0] != bc1.split('_', 1)[0])
            ):
                bc1_df = sub_df.query('binner_config == @bc1')
                print('Comparing:', bc0, 'and', bc1)
                wc_stat = sci_stats.wilcoxon(bc0_df['ext_nc_uniq'],
                                             bc1_df['ext_nc_uniq']
                                             )
                print(wc_stat)
    flurp
    '''

    count_nc_df = sub_df.groupby(['binner_config']
                                 )['ext_nc_uniq'].sum().reset_index()
    sorted_nc_df = count_nc_df.sort_values(by=['ext_nc_uniq'], ascending=False
                                           ).reset_index()
    sorted_nc_df['level_mode'] = level_mode
    cnt_df_list.append(sorted_nc_df)
    print(sorted_nc_df)

cat_cnt_df = pd.concat(cnt_df_list)
cat_cnt_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                        if 'SABer' in x else x.split('_', 1)[0]
                        for x in cat_cnt_df['binner_config']
                        ]

filter_list = ['SABer_denovo', 'SABer_hdbscan', 'SABer_ocsvm', 'SABer_intersect']
filter_cnt_df = cat_cnt_df.query("binner not in @filter_list")
dedup_cnt_df = filter_cnt_df.drop_duplicates(subset=['binner', 'level_mode'])
dedup_cnt_df['binner_config_level_mode'] = [x + '_' + y for x, y
                                            in zip(dedup_cnt_df['binner_config'],
                                                   dedup_cnt_df['level_mode']
                                                   )]
dedup_cnt_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.NC.uniq_sample.counts.tsv'),
                    sep='\t', index=False
                    )

# Calculate the Recall diff between SAGs and xPGs
bclm_list = list(dedup_cnt_df['binner_config_level_mode'].unique())
bin_cat_df['binner_config_level_mode'] = [x + '_' + y for x, y
                                          in zip(bin_cat_df['binner_config'],
                                                 bin_cat_df['level_mode']
                                                 )]
xpg_keep_list = ['best_label', 'exact_label',
                 'precision', 'sensitivity', 'MCC',
                 'dataset', 'sample_type', 'sample_id',
                 'mode', 'param_set'
                 ]
bin_cat_df.rename(columns={">20Kb": "over20Kb"}, inplace=True)
bin_cat_df['bin_bp'] = bin_cat_df['TP'] + bin_cat_df['FP']

# output best config P, R, and MCC to table
metric_df = bin_cat_df.query("level == 'strain_absolute' & "
                             "NC_bins == 'Yes' & "
                             "MQ_bins == 'Yes' & "
                             "over20Kb == 'Yes' & "
                             "binner_config_level_mode in @bclm_list"
                             )
metric_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.NC.metrics.tsv'),
                 sep='\t', index=False
                 )

xpg_single_df = bin_cat_df.query("algorithm == 'xPG' & "
                                 "level == 'strain_absolute' & "
                                 "bin_mode == 'single' & "
                                 "NC_bins == 'Yes' & "
                                 "MQ_bins == 'Yes' & "
                                 "over20Kb == 'Yes' & "
                                 "binner_config_level_mode in @bclm_list"
                                 )[xpg_keep_list]

xpg_single_df['ref_id'] = [x.rsplit('.', 1)[0]
                           for x in xpg_single_df['best_label']
                           ]
diffxpg_single_df = xpg_single_df.merge(diffdna_single_df,
                                        on=['ref_id', 'sample_type',
                                            'sample_id', 'mode',
                                            'param_set'],
                                        how='left'
                                        )

diff_filter_list = ['best_label', 'dataset', 'sample_type',
                    'sample_id', 'mode', 'param_set'
                    ]
aln_single_df = diffxpg_single_df.pivot_table(values='AlignedBases',
                                              index=diff_filter_list,
                                              columns='tag'
                                              ).reset_index()
aln_single_df.columns = ['best_label', 'dataset', 'sample_type',
                         'sample_id', 'mode', 'param_set',
                         'AlignedBases_SAG', 'AlignedBases_xPG'
                         ]
tot_single_df = diffxpg_single_df.pivot(values='TotalBases',
                                        index=diff_filter_list,
                                        columns='tag'
                                        ).reset_index()
tot_single_df.columns = ['best_label', 'dataset', 'sample_type',
                         'sample_id', 'mode', 'param_set',
                         'TotalBases_SAG', 'TotalBases_xPG'
                         ]
unaln_single_df = diffxpg_single_df.pivot(values='UnalignedBases',
                                          index=diff_filter_list,
                                          columns='tag'
                                          ).reset_index()
unaln_single_df.columns = ['best_label', 'dataset', 'sample_type',
                           'sample_id', 'mode', 'param_set',
                           'UnalignedBases_SAG', 'UnalignedBases_xPG'
                           ]

df_list = [xpg_single_df, aln_single_df,
           tot_single_df, unaln_single_df
           ]
sagxpg_single_df = reduce(lambda left, right:
                          pd.merge(left, right,
                                   on=diff_filter_list,
                                   how='left'), df_list
                          )
sagxpg_single_df['xPG'] = sagxpg_single_df['AlignedBases_xPG'] / \
                          sagxpg_single_df['TotalBases_xPG']
sagxpg_single_df['SAG'] = sagxpg_single_df['AlignedBases_SAG'] / \
                          sagxpg_single_df['TotalBases_SAG']
sagxpg_single_df['F1_SAG'] = [calc_f1score(1.0, x) for x in sagxpg_single_df['SAG']]
sagxpg_single_df['F1_xPG'] = [calc_f1score(x, y) for x, y in
                              zip(sagxpg_single_df['precision'],
                                  sagxpg_single_df['sensitivity']
                                  )]
sagxpg_single_df['percent_increase'] = ((sagxpg_single_df['AlignedBases_xPG'] -
                                         sagxpg_single_df['AlignedBases_SAG']) /
                                        sagxpg_single_df['AlignedBases_SAG']
                                        ) * 100
val_list = ['xPG', 'SAG']
R_df = pd.melt(sagxpg_single_df,
               id_vars=diff_filter_list,
               value_vars=val_list
               )
R_df.columns = ['best_label', 'dataset', 'sample_type',
                'sample_id', 'mode', 'param_set',
                'data_type', 'recall'
                ]
R_df['type_rank'] = [type2rank[x] for x in R_df['dataset']]
R_df.sort_values(by=['data_type', 'type_rank'], inplace=True)
R_df.to_csv(os.path.join(workdir, 'tables/SABer.SAG_xPG.NC.tsv'), sep='\t', index=False)
palette_map = {'xPG': cmap_muted[1], 'SAG': cmap_muted[7]}
boxie = sns.catplot(x="dataset", y="recall", hue="data_type",
                    col='mode', row='param_set',
                    alpha=0.50, jitter=0.25,
                    data=R_df, palette=palette_map
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/SABer.SAG_xPG.NC.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# keep_level = ['exact_absolute', 'strain_absolute']
keep_level = ['strain_absolute']
temp_cat_df = sample_metrics_df.copy().query("level in @keep_level")
temp_cat_df['binner_config_level_mode'] = [x + '_' + y for x, y
                                           in zip(temp_cat_df['binner_config'],
                                                  temp_cat_df['level_mode']
                                                  )]
temp_cat_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                         if 'SABer' in x else x.split('_', 1)[0]
                         for x in temp_cat_df['binner_config']
                         ]
temp_filter_df = temp_cat_df.query("binner not in @filter_list")
sub_binstat_df = temp_filter_df.query("binner_config_level_mode in @bclm_list")
sub_binstat_df['bin_rank'] = [binner2rank[x] for x in
                              sub_binstat_df['binner']
                              ]
sub_binstat_df['type_rank'] = [type2rank[x] for x in
                               sub_binstat_df['dataset']
                               ]
sub_binstat_df['level_rank'] = [level2rank[x] for x in
                                sub_binstat_df['level_mode']
                                ]
sub_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'
                               ], inplace=True
                           )

# Boxplots for mode and param set
boxie = sns.catplot(x="dataset", y="ext_nc_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.NC.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Boxplots for precision
boxie = sns.catplot(x="dataset", y="nc_avg_p", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.NC_P.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Boxplots for recall
boxie = sns.catplot(x="dataset", y="nc_avg_r", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.NC_R.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Boxplots for mcc
boxie = sns.catplot(x="dataset", y="nc_avg_mcc", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.NC_MCC.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# keep_level = ['exact_absolute', 'strain_absolute']
keep_level = ['strain_absolute']
temp_cat_df = dataset_metrics_df.copy().query("level in @keep_level")
temp_cat_df['binner_config_level_mode'] = [x + '_' + y for x, y
                                           in zip(temp_cat_df['binner_config'],
                                                  temp_cat_df['level_mode']
                                                  )]
temp_cat_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                         if 'SABer' in x else x.split('_', 1)[0]
                         for x in temp_cat_df['binner_config']
                         ]
temp_filter_df = temp_cat_df.query("binner not in @filter_list")
sub_binstat_df = temp_filter_df.query("binner_config_level_mode in @bclm_list")
sub_binstat_df['bin_rank'] = [binner2rank[x] for x in
                              sub_binstat_df['binner']
                              ]
sub_binstat_df['type_rank'] = [type2rank[x] for x in
                               sub_binstat_df['dataset']
                               ]
sub_binstat_df['level_rank'] = [level2rank[x] for x in
                                sub_binstat_df['level_mode']
                                ]
sub_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'
                               ], inplace=True
                           )
# Barplots for mode and param set
sum_binstat_df = sub_binstat_df.groupby(['binner', 'bin_rank',
                                         'type_rank', 'level_rank',
                                         'level_mode', 'dataset',
                                         'binner_config_level_mode']
                                        )['ext_nc_uniq'].sum().reset_index()
sum_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'],
                           inplace=True)

barie = sns.catplot(x="dataset", y="ext_nc_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="bar", data=sum_binstat_df,
                    linewidth=0.75, saturation=0.75,
                    palette=binner2cmap
                    )
barie.savefig(os.path.join(workdir, 'barplots/ALL_BINNERS.NC.barplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

sum_binstat_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.NC.uniq_dataset.counts.tsv'),
                      sep='\t', index=False
                      )

########################################################################################################################
##### Scatter of >200Kbp bins ##########################################################################################
########################################################################################################################
# output best config P, R, and MCC to table
bin_cat_df['bin_bp'] = bin_cat_df['TP'] + bin_cat_df['FP']
size_filter_df = bin_cat_df.query("level == 'strain_absolute' & "
                                  "bin_bp >= 200000 & "
                                  "binner_config_level_mode in @bclm_list"
                                  )
size_filter_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                            if 'SABer' in x else x.split('_', 1)[0]
                            for x in size_filter_df['binner_config']
                            ]
indices = ['binner', 'bin_mode']
cols = ['precision', 'sensitivity', 'MCC', 'F1']
piv_metric_df = pd.melt(size_filter_df, id_vars=indices,
                        value_vars=cols, var_name='metric',
                        value_name='value'
                        )
# Boxplots of metrics for bins >=200Kbp
boxie = sns.catplot(x="metric", y="value", hue="binner",
                    col="bin_mode", col_wrap=2,
                    kind="box", data=piv_metric_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.200Kbp.boxplot.png'),
              dpi=300)

plt.clf()
plt.close()
print(size_filter_df.head())

########################################################################################################################
##### RUN MQ STATS #####################################################################################################
########################################################################################################################
print('############################################################')
print("RUN MQ STATS")
print('############################################################')
cnt_df_list = []
for level_mode in sample_metrics_df['level_mode'].unique():
    print('############################################################')
    print(f"The Level tested is {level_mode}")
    print('############################################################')
    sub_df = sample_metrics_df.query("level_mode == @level_mode")
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = sci_stats.f_oneway(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_mq_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    m_comp = pairwise_tukeyhsd(endog=sub_df['ext_mq_uniq'],
                               groups=sub_df['binner_config'],
                               alpha=0.05
                               )
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    stat, p = sci_stats.kruskal(
        *(sub_df.loc[sub_df['binner_config'] == group, 'ext_mq_uniq']
          for group in sub_df['binner_config'].unique()
          ))
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    count_nc_df = sub_df.groupby(['binner_config']
                                 )['ext_mq_uniq'].sum().reset_index()
    sorted_nc_df = count_nc_df.sort_values(by=['ext_mq_uniq'], ascending=False
                                           ).reset_index()
    sorted_nc_df['level_mode'] = level_mode
    cnt_df_list.append(sorted_nc_df)
    print(sorted_nc_df)

cat_cnt_df = pd.concat(cnt_df_list)
cat_cnt_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                        if 'SABer' in x else x.split('_', 1)[0]
                        for x in cat_cnt_df['binner_config']
                        ]
filter_list = ['SABer_denovo', 'SABer_hdbscan', 'SABer_ocsvm', 'SABer_intersect']
filter_cnt_df = cat_cnt_df.query("binner not in @filter_list")
dedup_cnt_df = filter_cnt_df.drop_duplicates(subset=['binner', 'level_mode'])
dedup_cnt_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.MQ.uniq_sample.counts.tsv'),
                    sep='\t', index=False
                    )
keep_binners_list = list(dedup_cnt_df['binner_config'])
keep_levmod_list = list(dedup_cnt_df['level_mode'])
# keep_level = ['exact_absolute', 'strain_absolute']
keep_level = ['strain_absolute']
temp_cat_df = sample_metrics_df.copy().query("level in @keep_level")
temp_cat_df['binner_config_level_mode'] = [x + '_' + y for x, y
                                           in zip(temp_cat_df['binner_config'],
                                                  temp_cat_df['level_mode']
                                                  )]
temp_cat_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                         if 'SABer' in x else x.split('_', 1)[0]
                         for x in temp_cat_df['binner_config']
                         ]
temp_filter_df = temp_cat_df.query("binner not in @filter_list")
sub_binstat_df = temp_filter_df.query("binner_config_level_mode in @bclm_list")
sub_binstat_df['bin_rank'] = [binner2rank[x] for x in
                              sub_binstat_df['binner']
                              ]
sub_binstat_df['type_rank'] = [type2rank[x] for x in
                               sub_binstat_df['dataset']
                               ]
sub_binstat_df['level_rank'] = [level2rank[x] for x in
                                sub_binstat_df['level_mode']
                                ]
sub_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'
                               ], inplace=True
                           )
# Boxplots for mode and param set
boxie = sns.catplot(x="dataset", y="ext_mq_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.MQ.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Boxplots for precision
boxie = sns.catplot(x="dataset", y="mq_avg_p", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.MQ_P.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Boxplots for recall
boxie = sns.catplot(x="dataset", y="mq_avg_r", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.MQ_R.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# Boxplots for mcc
boxie = sns.catplot(x="dataset", y="mq_avg_mcc", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="box", data=sub_binstat_df, notch=True,
                    linewidth=0.75, saturation=0.75, width=0.75,
                    palette=binner2cmap
                    )
boxie.savefig(os.path.join(workdir, 'boxplots/ALL_BINNERS.MQ_MCC.boxplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

# keep_level = ['exact_absolute', 'strain_absolute']
keep_level = ['strain_absolute']
temp_cat_df = dataset_metrics_df.copy().query("level in @keep_level")
temp_cat_df['binner_config_level_mode'] = [x + '_' + y for x, y
                                           in zip(temp_cat_df['binner_config'],
                                                  temp_cat_df['level_mode']
                                                  )]
temp_cat_df['binner'] = [x.split('_', 2)[0] + '_' + x.split('_', 2)[1]
                         if 'SABer' in x else x.split('_', 1)[0]
                         for x in temp_cat_df['binner_config']
                         ]
temp_filter_df = temp_cat_df.query("binner not in @filter_list")
sub_binstat_df = temp_filter_df.query("binner_config_level_mode in @bclm_list")
sub_binstat_df['bin_rank'] = [binner2rank[x] for x in
                              sub_binstat_df['binner']
                              ]
sub_binstat_df['type_rank'] = [type2rank[x] for x in
                               sub_binstat_df['dataset']
                               ]
sub_binstat_df['level_rank'] = [level2rank[x] for x in
                                sub_binstat_df['level_mode']
                                ]
sub_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'
                               ], inplace=True
                           )
# Barplots for mode and param set
sum_binstat_df = sub_binstat_df.groupby(['binner', 'bin_rank',
                                         'type_rank', 'level_rank',
                                         'level_mode', 'dataset',
                                         'binner_config_level_mode']
                                        )['ext_mq_uniq'].sum().reset_index()
sum_binstat_df.sort_values(by=['level_rank', 'bin_rank', 'type_rank'],
                           inplace=True)
barie = sns.catplot(x="dataset", y="ext_mq_uniq", hue="binner",
                    col="level_mode", col_wrap=2,
                    kind="bar", data=sum_binstat_df,
                    linewidth=0.75, saturation=0.75,
                    palette=binner2cmap
                    )
barie.savefig(os.path.join(workdir, 'barplots/ALL_BINNERS.MQ.barplot.png'),
              dpi=300
              )
plt.clf()
plt.close()

sum_binstat_df.to_csv(os.path.join(workdir, 'tables/ALL_BINNERS.MQ.uniq_dataset.counts.tsv'),
                      sep='\t', index=False
                      )

sys.exit()
########################################################################################################################
##### OLD CODE BELOW ###################################################################################################
########################################################################################################################


########################################################################################################################
# Single SABer - Near Complete, Absolute
########################################################################################################################
print('############################################################')
print('# Single SABer')
print('############################################################')

ss_df['label'] = [type2label[x] for x in ss_df['sample_type']]
ss_df['algo_rank'] = [algo2rank[x] for x in ss_df['algorithm']]
ss_df['type_rank'] = [type2rank[x] for x in ss_df['sample_type']]
ss_df['mode_rank'] = [mode2rank[x] for x in ss_df['mode']]
ss_df['param_rank'] = [param2rank[x] for x in ss_df['param_set']]
ss_df['mode_paramset'] = [str(x) + '_' + str(y) for x, y in
                          zip(ss_df['mode'], ss_df['param_set'])
                          ]
ss_df['algo_param'] = ['_'.join([str(x), str(y)]) for x, y in
                       zip(ss_df['algorithm'], ss_df['param_set'])
                       ]
ss_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(ss_df['label'], ss_df['sample_id'])
                         ]
ss_abs_str_df = ss_df.query("level == 'strain_absolute'")
ss_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
ss_abs_str_median_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in ss_abs_str_df['algo_param'].unique():
    sub_ss_df = ss_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_nc_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_nc_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_nc_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_nc_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_nc_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(ss_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
                     col="param_set", col_wrap=2,
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.NC.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.NC.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_nc_uniq", hue="algorithm",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.NC.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
ss_abs_str_median_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = ss_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in ss_abs_str_df['algo_param'].unique():
    sub_ss_df = ss_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_mq_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_mq_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_mq_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_mq_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_mq_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(ss_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
                     col="param_set", col_wrap=2,
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.MQ.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.MQ.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_mq_uniq", hue="algorithm",
                     kind="box", data=ss_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.single.absolute.MQ.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

########################################################################################################################
# Multi SABer
########################################################################################################################
print('############################################################')
print('# Multi SABer')
print('############################################################')

sm_df = pd.read_csv(saber_multi_file, header=0, sep='\t')
sm_df['label'] = [type2label[x] for x in sm_df['sample_type']]
sm_df['algo_rank'] = [algo2rank[x] for x in sm_df['algorithm']]
sm_df['type_rank'] = [type2rank[x] for x in sm_df['sample_type']]
sm_df['mode_rank'] = [mode2rank[x] for x in sm_df['mode']]
sm_df['param_rank'] = [param2rank[x] for x in sm_df['param_set']]
sm_df['mode_paramset'] = [str(x) + '_' + str(y) for x, y in
                          zip(sm_df['mode'], sm_df['param_set'])
                          ]
sm_df['algo_param'] = ['_'.join([str(x), str(y)]) for x, y in
                       zip(sm_df['algorithm'], sm_df['param_set'])
                       ]
sm_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(sm_df['label'], sm_df['sample_id'])
                         ]
sm_abs_str_df = sm_df.query("level == 'strain_absolute'")
sm_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'mode_rank', 'param_set'
                              ], inplace=True)
ss_abs_str_median_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in sm_abs_str_df['algo_param'].unique():
    sub_ss_df = sm_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_nc_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_nc_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_nc_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_nc_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_nc_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(sm_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
                     col="param_set", col_wrap=2,
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.NC.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_nc_uniq", hue="algorithm",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.NC.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_nc_uniq", hue="algorithm",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.NC.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
ss_abs_str_median_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
ss_abs_str_median_df.columns = ['mode', 'algorithm', 'median']
ss_abs_str_mean_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
ss_abs_str_mean_df.columns = ['mode', 'algorithm', 'mean']
ss_abs_str_std_df = sm_abs_str_df.groupby(['mode', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
ss_abs_str_std_df.columns = ['mode', 'algorithm', 'stdev']
stats_df_list = [ss_abs_str_median_df, ss_abs_str_mean_df, ss_abs_str_std_df]
ss_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['mode', 'algorithm']), stats_df_list)
ss_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(ss_abs_str_stats_df)

MR_count = 0
BC_count = 0
BM_count = 0
for algo_param in sm_abs_str_df['algo_param'].unique():
    sub_ss_df = sm_abs_str_df.query("algo_param == @algo_param")
    mr_count = sub_ss_df.query("mode == 'majority_rule'")['ext_mq_uniq'].sum()
    bc_count = sub_ss_df.query("mode == 'best_cluster'")['ext_mq_uniq'].sum()
    bm_count = sub_ss_df.query("mode == 'best_match'")['ext_mq_uniq'].sum()
    test_df = sub_ss_df.pivot(index='label_sample', columns='mode', values='ext_mq_uniq')

    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(test_df['majority_rule'],
                                    test_df['best_cluster'],
                                    test_df['best_match']
                                    )
    m_comp = pairwise_tukeyhsd(endog=sub_ss_df['ext_mq_uniq'], groups=sub_ss_df['mode'], alpha=0.05)
    stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

    print(f"\nThe Algorithm tested is {algo_param}")
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
    print(f"\nResults of Tukey HSD test:")
    print(m_comp)
    print(f"\nResults of Wilcoxon Signed-Rank Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    stat, p = kruskal(test_df['majority_rule'],
                      test_df['best_cluster'],
                      test_df['best_match']
                      )
    print(f"\nResults of Kruskal-Wallis H Test:")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print(f"\nThe total number of NC bins for each mode is:\n")
    print(f"\t Majority Rule: {mr_count}")
    print(f"\t Best Cluster: {bc_count}")
    print(f"\t Best Match: {bm_count}")
    MR_count += mr_count
    BC_count += bc_count
    BM_count += bm_count

print(f"Total Majority Rule: {MR_count}")
print(f"Total Best Cluster: {BC_count}")
print(f"Total Best Match: {BM_count}")
lengs = len(sm_abs_str_df['algo_param'].unique())
print(f"Average Majority Rule: {MR_count / lengs}")
print(f"Average Best Cluster: {BC_count / lengs}")
print(f"Average Best Match: {BM_count / lengs}")

# Boxplots for mode and param set
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
                     col="param_set", col_wrap=2,
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.mode_param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for mode
ss_box = sns.catplot(x="mode", y="ext_mq_uniq", hue="algorithm",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.mode.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Boxplots for param set
ss_box = sns.catplot(x="param_set", y="ext_mq_uniq", hue="algorithm",
                     kind="box", data=sm_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
ss_box.savefig(os.path.join(workdir, 'SABer.multi.absolute.MQ.param.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

########################################################################################################################
# Single UniteM
########################################################################################################################
print('############################################################')
print('# Single UniteM')
print('############################################################')

us_df = pd.read_csv(unitem_single_file, header=0, sep='\t')
print(us_df.head())
print(us_df.columns)
us_df['label'] = [type2label[x] for x in us_df['sample_type']]
us_df['algo_rank'] = [algo2rank[x] for x in us_df['algorithm']]
us_df['type_rank'] = [type2rank[x] for x in us_df['sample_type']]
us_df['binner_rank'] = [binner2rank[x] for x in us_df['binner']]
us_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(us_df['label'], us_df['sample_id'])
                         ]
us_abs_str_df = us_df.query("level == 'strain_absolute'")
us_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'binner_rank'
                              ], inplace=True)
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df,
                 us_abs_str_std_df
                 ]
us_abs_str_stats_df = reduce(lambda x, y:
                             pd.merge(x, y, on=['binner', 'algorithm']),
                             stats_df_list
                             )
us_abs_str_stats_df.sort_values(by='mean', ascending=False,
                                inplace=True
                                )

print(us_abs_str_stats_df)

test_df = us_abs_str_df.pivot(index='label_sample', columns='binner', values='ext_nc_uniq')
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(test_df['maxbin_ms40'],
                                test_df['maxbin_ms107'],
                                test_df['metabat_specific'],
                                test_df['metabat_veryspecific'],
                                test_df['metabat_superspecific'],
                                test_df['metabat_sensitive'],
                                test_df['metabat_verysensitive'],
                                test_df['metabat2']
                                )
m_comp = pairwise_tukeyhsd(endog=us_abs_str_df['ext_nc_uniq'], groups=us_abs_str_df['binner'], alpha=0.05)
# stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

print(f"\nThe Algorithm tested is UniteM Binners")
print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
print(f"\nResults of Tukey HSD test:")
print(m_comp)
# print(f"\nResults of Wilcoxon Signed-Rank Test:")
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
# alpha = 0.05
# if p > alpha:
#    print('Same distribution (fail to reject H0)')
# else:
#    print('Different distribution (reject H0)')
stat, p = kruskal(test_df['maxbin_ms40'],
                  test_df['maxbin_ms107'],
                  test_df['metabat_specific'],
                  test_df['metabat_veryspecific'],
                  test_df['metabat_superspecific'],
                  test_df['metabat_sensitive'],
                  test_df['metabat_verysensitive'],
                  test_df['metabat2']
                  )
print(f"\nResults of Kruskal-Wallis H Test:")
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
print(f"\nThe total number of NC bins for each binner is:\n")
print(us_abs_str_df.groupby(['binner'])['ext_nc_uniq'].sum())

# Boxplots for binner and param set
us_box = sns.catplot(x="label", y="ext_nc_uniq", hue="binner",
                     kind="box", data=us_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
us_box.savefig(os.path.join(workdir, 'UniteM.single.absolute.NC.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df, us_abs_str_std_df]
us_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['binner', 'algorithm']), stats_df_list)
us_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(us_abs_str_stats_df)

test_df = us_abs_str_df.pivot(index='label_sample', columns='binner', values='ext_mq_uniq')
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(test_df['maxbin_ms40'],
                                test_df['maxbin_ms107'],
                                test_df['metabat_specific'],
                                test_df['metabat_veryspecific'],
                                test_df['metabat_superspecific'],
                                test_df['metabat_sensitive'],
                                test_df['metabat_verysensitive'],
                                test_df['metabat2']
                                )
m_comp = pairwise_tukeyhsd(endog=us_abs_str_df['ext_mq_uniq'], groups=us_abs_str_df['binner'], alpha=0.05)
# stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

print(f"\nThe Algorithm tested is UniteM Binners")
print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
print(f"\nResults of Tukey HSD test:")
print(m_comp)
# print(f"\nResults of Wilcoxon Signed-Rank Test:")
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
# alpha = 0.05
# if p > alpha:
#    print('Same distribution (fail to reject H0)')
# else:
#    print('Different distribution (reject H0)')
stat, p = kruskal(test_df['maxbin_ms40'],
                  test_df['maxbin_ms107'],
                  test_df['metabat_specific'],
                  test_df['metabat_veryspecific'],
                  test_df['metabat_superspecific'],
                  test_df['metabat_sensitive'],
                  test_df['metabat_verysensitive'],
                  test_df['metabat2']
                  )
print(f"\nResults of Kruskal-Wallis H Test:")
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
print(f"\nThe total number of NC bins for each binner is:\n")
print(us_abs_str_df.groupby(['binner'])['ext_mq_uniq'].sum())

# Boxplots for binner and param set
us_box = sns.catplot(x="label", y="ext_mq_uniq", hue="binner",
                     kind="box", data=us_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
us_box.savefig(os.path.join(workdir, 'UniteM.single.absolute.MQ.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

########################################################################################################################
# Multi UniteM
########################################################################################################################
print('############################################################')
print('# Multi UniteM')
print('############################################################')

us_df = pd.read_csv(unitem_multi_file, header=0, sep='\t')
print(us_df.head())
print(us_df.columns)
us_df['label'] = [type2label[x] for x in us_df['sample_type']]
us_df['algo_rank'] = [algo2rank[x] for x in us_df['algorithm']]
us_df['type_rank'] = [type2rank[x] for x in us_df['sample_type']]
us_df['binner_rank'] = [binner2rank[x] for x in us_df['binner']]
us_df['label_sample'] = ['_'.join([str(x), str(y)]) for x, y in
                         zip(us_df['label'], us_df['sample_id'])
                         ]
us_abs_str_df = us_df.query("level == 'strain_absolute'")
us_abs_str_df.sort_values(by=['type_rank', 'algo_rank',
                              'binner_rank'
                              ], inplace=True)
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_nc_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_nc_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_nc_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df,
                 us_abs_str_std_df
                 ]
us_abs_str_stats_df = reduce(lambda x, y:
                             pd.merge(x, y, on=['binner', 'algorithm']),
                             stats_df_list
                             )
us_abs_str_stats_df.sort_values(by='mean', ascending=False,
                                inplace=True
                                )

print(us_abs_str_stats_df)

test_df = us_abs_str_df.pivot(index='label_sample', columns='binner', values='ext_nc_uniq')
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(test_df['maxbin_ms40'],
                                test_df['maxbin_ms107'],
                                test_df['metabat_specific'],
                                test_df['metabat_veryspecific'],
                                test_df['metabat_superspecific'],
                                test_df['metabat_sensitive'],
                                test_df['metabat_verysensitive'],
                                test_df['metabat2']
                                )
m_comp = pairwise_tukeyhsd(endog=us_abs_str_df['ext_nc_uniq'], groups=us_abs_str_df['binner'], alpha=0.05)
# stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

print(f"\nThe Algorithm tested is UniteM Binners")
print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
print(f"\nResults of Tukey HSD test:")
print(m_comp)
# print(f"\nResults of Wilcoxon Signed-Rank Test:")
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
# alpha = 0.05
# if p > alpha:
#    print('Same distribution (fail to reject H0)')
# else:
#    print('Different distribution (reject H0)')
stat, p = kruskal(test_df['maxbin_ms40'],
                  test_df['maxbin_ms107'],
                  test_df['metabat_specific'],
                  test_df['metabat_veryspecific'],
                  test_df['metabat_superspecific'],
                  test_df['metabat_sensitive'],
                  test_df['metabat_verysensitive'],
                  test_df['metabat2']
                  )
print(f"\nResults of Kruskal-Wallis H Test:")
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
print(f"\nThe total number of NC bins for each binner is:\n")
print(us_abs_str_df.groupby(['binner'])['ext_nc_uniq'].sum())

# Boxplots for binner and param set
us_box = sns.catplot(x="label", y="ext_nc_uniq", hue="binner",
                     kind="box", data=us_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
us_box.savefig(os.path.join(workdir, 'UniteM.multi.absolute.NC.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

# Medium Quality
us_abs_str_median_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                             )[['ext_mq_uniq'
                                                ]].median().reset_index()
us_abs_str_median_df.columns = ['binner', 'algorithm', 'median']
us_abs_str_mean_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                           )[['ext_mq_uniq'
                                              ]].mean().reset_index()
us_abs_str_mean_df.columns = ['binner', 'algorithm', 'mean']
us_abs_str_std_df = us_abs_str_df.groupby(['binner', 'algorithm']
                                          )[['ext_mq_uniq'
                                             ]].std().reset_index()
us_abs_str_std_df.columns = ['binner', 'algorithm', 'stdev']
stats_df_list = [us_abs_str_median_df, us_abs_str_mean_df, us_abs_str_std_df]
us_abs_str_stats_df = reduce(lambda x, y: pd.merge(x, y, on=['binner', 'algorithm']), stats_df_list)
us_abs_str_stats_df.sort_values(by='mean', ascending=False, inplace=True)

print(us_abs_str_stats_df)

test_df = us_abs_str_df.pivot(index='label_sample', columns='binner', values='ext_mq_uniq')
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(test_df['maxbin_ms40'],
                                test_df['maxbin_ms107'],
                                test_df['metabat_specific'],
                                test_df['metabat_veryspecific'],
                                test_df['metabat_superspecific'],
                                test_df['metabat_sensitive'],
                                test_df['metabat_verysensitive'],
                                test_df['metabat2']
                                )
m_comp = pairwise_tukeyhsd(endog=us_abs_str_df['ext_mq_uniq'], groups=us_abs_str_df['binner'], alpha=0.05)
# stat, p = wilcoxon(test_df['majority_rule'], test_df['best_cluster'])

print(f"\nThe Algorithm tested is UniteM Binners")
print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")
print(f"\nResults of Tukey HSD test:")
print(m_comp)
# print(f"\nResults of Wilcoxon Signed-Rank Test:")
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
# alpha = 0.05
# if p > alpha:
#    print('Same distribution (fail to reject H0)')
# else:
#    print('Different distribution (reject H0)')
stat, p = kruskal(test_df['maxbin_ms40'],
                  test_df['maxbin_ms107'],
                  test_df['metabat_specific'],
                  test_df['metabat_veryspecific'],
                  test_df['metabat_superspecific'],
                  test_df['metabat_sensitive'],
                  test_df['metabat_verysensitive'],
                  test_df['metabat2']
                  )
print(f"\nResults of Kruskal-Wallis H Test:")
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
print(f"\nThe total number of NC bins for each binner is:\n")
print(us_abs_str_df.groupby(['binner'])['ext_mq_uniq'].sum())

# Boxplots for binner and param set
us_box = sns.catplot(x="label", y="ext_mq_uniq", hue="binner",
                     kind="box", data=us_abs_str_df, notch=True,
                     linewidth=0.75, saturation=0.75, width=0.75,
                     palette=sns.color_palette("muted")
                     )
us_box.savefig(os.path.join(workdir, 'UniteM.multi.absolute.MQ.binner.boxplot.png'),
               dpi=300
               )
plt.clf()
plt.close()

