import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import numpy as np
import sys
import os
from os import listdir, makedirs, path


sns.set_context("poster")
sns.set_style('whitegrid')
sns.set(font_scale=0.75)
err_path = sys.argv[1]

algo_path = err_path + 'multi-algo'
if not path.exists(algo_path):
    makedirs(algo_path)
level_path = err_path + 'multi-level'
if not path.exists(level_path):
    makedirs(level_path)


err_file = err_path + 'All_stats_count.tsv'
err_df = pd.read_csv(err_file, header=0, sep='\t')
map_algo = {'synSAG':'synSAG', 'MinHash':'MinHash', 'TPM':'TPM', 'tetra_gmm':'GMM',
            'tetra_svm':'OCSVM', 'tetra_iso':'Isolation Forest', 'tetra_comb':'Tetra Ensemble',
            'gmm_combined':'Final GMM', 'svm_combined':'Final OCSVM',
            'iso_combined':'Final Isolation Forest', 'comb_combined':'Final Ensemble',
            'gmm_extend': 'SABer-GMM', 'svm_extend': 'SABer-OCSVM',
            'iso_extend': 'SABer-Isolation Forest', 'comb_extend': 'SABer-Ensemble'
            }
err_df['algorithm'] = [map_algo[x] for x in err_df['algorithm']]
err_df['level'] = ['exact' if x == 'perfect' else x for x in err_df['level']]
err_trim_df = err_df.loc[(err_df['statistic'] != 'F1_score') & (err_df['score'] > 0)]

unstack_df = err_trim_df.set_index(['sag_id', 'algorithm', 'level', 'statistic']
                                    ).unstack('statistic'
                                    )
unstack_df.reset_index(inplace=True)
unstack_df.columns = ['sag_id', 'algorithm', 'level', 'MCC', 'Precision',
                        'Sensitivity'
                        ]

val_df_list = []
outlier_list = []
for algo in set(unstack_df['algorithm']):
    algo_df = unstack_df.loc[unstack_df['algorithm'] == algo]
    for level in set(unstack_df['level']):
        level_df = algo_df.loc[algo_df['level'] == level].set_index(
                                                ['sag_id', 'algorithm', 'level']
                                                )
        for stat in ['MCC', 'Precision', 'Sensitivity']:
            stat_df = level_df[[stat]]
            mean = list(stat_df.mean())[0]
            var = list(stat_df.var())[0]
            skew = list(stat_df.skew())[0]
            kurt = list(stat_df.kurt())[0]
            IQ_25 = list(stat_df.quantile(0.25))[0]
            IQ_75 = list(stat_df.quantile(0.75))[0]
            IQ_10 = list(stat_df.quantile(0.10))[0]
            IQ_90 = list(stat_df.quantile(0.90))[0]
            IQ_05 = list(stat_df.quantile(0.05))[0]
            IQ_95 = list(stat_df.quantile(0.95))[0]
            IQ_01 = list(stat_df.quantile(0.01))[0]
            IQ_99 = list(stat_df.quantile(0.99))[0]
            IQR = IQ_75 - IQ_25
            # calc Tukey Fences
            upper_bound = IQ_75 + (1.5 * IQR)
            lower_bound = IQ_25 - (1.5 * IQR)
            header_list = ['algorithm', 'level', 'stat', 'mean', 'var', 'skew', 'kurt',
                            'IQ_25', 'IQ_75', 'IQ_10', 'IQ_90', 'IQ_05', 'IQ_95',
                            'IQ_01', 'IQ_99', 'IQR (25-75)', 'upper_bound', 'lower_bound'
                            ]
            val_list = [algo, level, stat, mean, var, skew, kurt, IQ_25, IQ_75,
                        IQ_10, IQ_90, IQ_05, IQ_95, IQ_01, IQ_99, IQR, upper_bound,
                        lower_bound
                        ]
            val_df =pd.DataFrame([val_list], columns=header_list)
            val_df_list.append(val_df)
            stat_df['statistic'] = stat
            stat_df.reset_index(inplace=True)
            stat_df.columns = ['sag_id', 'algorithm', 'level', 'score', 'statistic']
            stat_df = stat_df[['sag_id', 'algorithm', 'level', 'statistic', 'score']]
            outlier_df = stat_df.loc[(stat_df['score'] < lower_bound) &
                                        (stat_df['score'] < 0.99)]
            outlier_list.append(outlier_df)

concat_val_df = pd.concat(val_df_list)
concat_val_df.to_csv(err_path + 'Compiled_stats.tsv', sep='\t', index=False)
concat_out_df = pd.concat(outlier_list)
concat_out_df.to_csv(err_path + 'Compiled_outliers.tsv', sep='\t', index=False)
level_order = ['domain', 'family', 'class', 'order', 'genus', 'species', 'strain', 'exact']
'''
g = sns.FacetGrid(unstack_df, col='level', row='algorithm', aspect=1.5,
                    col_order=level_order,
                    row_order=['mockSAG', 'MinHash', 'TPM', 'GMM', 'OCSVM',
                                'Isolation Forest', 'Tetra Ensemble', 'Final GMM',
                                'Final OCSVM', 'Final Isolation Forest', 'Final Ensemble',
                                'SABer-GMM', 'SABer-OCSVM', 'SABer-Isolation Forest',
                                'SABer-Ensemble'
                                ]
                    )
g = g.map(plt.scatter, 'Precision', 'Sensitivity', color='k', edgecolor='w')
g.savefig("precision_sensitivity_scatters.pdf", bbox_inches='tight')
plt.close()

g = sns.FacetGrid(unstack_df, col='level', row='algorithm', aspect=1.5, sharey=False,
                    sharex=False, xlim=[0, 1.1],
                    col_order=level_order,
                    row_order=['mockSAG', 'MinHash', 'TPM', 'GMM', 'OCSVM',
                                'Isolation Forest', 'Tetra Ensemble', 'Final GMM',
                                'Final OCSVM', 'Final Isolation Forest', 'Final Ensemble',
                                'SABer-GMM', 'SABer-OCSVM', 'SABer-Isolation Forest',
                                'SABer-Ensemble'
                                ]
                    )
g = g.map(plt.hist, 'Precision', bins=np.arange(0, 1.1, 0.02), color='k', edgecolor='w')
g.savefig("precision_histograms.pdf", bbox_inches='tight')
plt.close()

g = sns.FacetGrid(unstack_df, col='level', row='algorithm', aspect=1.5, sharey=False,
                    sharex=False, xlim=[0, 1.1],
                    col_order=level_order,
                    row_order=['mockSAG', 'MinHash', 'TPM', 'GMM', 'OCSVM',
                                'Isolation Forest', 'Tetra Ensemble', 'Final GMM',
                                'Final OCSVM', 'Final Isolation Forest', 'Final Ensemble',
                                'SABer-GMM', 'SABer-OCSVM', 'SABer-Isolation Forest',
                                'SABer-Ensemble'
                                ]
                    )
g = g.map(plt.hist, 'Sensitivity', bins=np.arange(0, 1.1, 0.02), color='k', edgecolor='w')
g.savefig("sensitivity_histograms.pdf", bbox_inches='tight')
plt.close()
'''
flierprops = dict(markerfacecolor='0.75', markersize=5, markeredgecolor='w',
              linestyle='none')

for level in set(err_trim_df['level']):
    level_df = err_trim_df.loc[err_trim_df['level'] == level]
    sns.set_context("paper")
    ax = sns.catplot(x="statistic", y="score", hue='algorithm', kind='box',
                        data=level_df, aspect=2, palette=sns.light_palette("black"),
                        flierprops=flierprops)
    plt.plot([-0.5, 3.5], [0.25, 0.25], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 3.5], [0.50, 0.50], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 3.5], [0.75, 0.75], linestyle='--', alpha=0.3, color='k')

    plt.ylim(0, 1)
    plt.xlim(-0.5, 3.5  )
    #plt.title('SAG-plus CAMI-1-High error analysis')
    ax._legend.set_title('Workflow\nStage')
    plt.savefig(algo_path + '/' + level + '_error_boxplox_count.png',
                bbox_inches='tight'
                )
    plt.clf()
    plt.close()

# build multi-level precision boxplot
stat_list = ['precision', 'sensitivity', 'MCC']
'''
mock_stat_df = err_df.loc[((err_df['algorithm'].isin(['mockSAG'])) &
                                    (err_df['level'].isin(['genus'])) &
                                    (err_df['statistic'].isin(stat_list))
                                    )]
mock_stat_df['level'] = 'mockSAG'
'''
for algo in set(err_trim_df['algorithm']):
    comb_stat_df = err_trim_df.loc[((err_trim_df['algorithm'] == algo) &
                                        (err_trim_df['level'].isin(level_order)) &
                                        (err_trim_df['statistic'].isin(stat_list))
                                        )]
    #concat_stat_df = pd.concat([mock_stat_df, comb_stat_df])
    sns.set_context("paper")
    ax = sns.catplot(x="level", y="score", hue='statistic', kind='box',
                        data=comb_stat_df, aspect=2, palette=sns.light_palette("black"),
                        flierprops=flierprops)

    plt.plot([-0.5, 4.5], [0.25, 0.25], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 4.5], [0.50, 0.50], linestyle='--', alpha=0.3, color='k')
    plt.plot([-0.5, 4.5], [0.75, 0.75], linestyle='--', alpha=0.3, color='k')

    #plt.ylim(0, 1)
    #plt.xlim(-0.5, 4.5)
    plt.title('SABer ' + algo + ' by Taxonomic-level')

    plt.savefig(level_path + '/' + algo.replace(' ', '_') + '_multi-level_boxplox_count.png',
                    bbox_inches='tight'
                    )
    plt.clf()
    plt.close()

# Stat by level line plot
err_deduped_df = err_trim_df.loc[err_trim_df['algorithm'].isin(['SABer-GMM', 'SABer-OCSVM',
                                                            'SABer-Isolation Forest',
                                                            'SABer-Ensemble'])
                                                            ]
'''
# AMBER Taxonomic
amber_tax_file = 'AMBER_taxonomic_results.tsv'
amber_tax_df = pd.read_csv(amber_tax_file, header=0, sep='\t')
amber_tax_df = amber_tax_df[['sample_id', 'Taxon ID', 'Scientific name', 'Taxonomic rank',
                                    'algorithm', 'Purity (bp)', 'Completeness (bp)'
                                    ]]
amber_tax_df.columns = ['sample_id', 'taxid', 'clade', 'level', 'algorithm',
                            'precision', 'sensitivity'
                            ]

amber_filter_df = amber_tax_df[['level', 'algorithm', 'precision', 'sensitivity']]
amber_filter_df = amber_filter_df[np.isfinite(amber_filter_df['precision'])]
amber_filter_df = amber_filter_df.loc[((amber_filter_df['precision'] != 0) &
                                        (amber_filter_df['sensitivity'] != 0)
                                        )]
#amber_filter_df = amber_filter_df.loc[((amber_filter_df['Purity (bp)'] != '') &
#                                        (amber_filter_df['Completeness (bp)'] != '') &
#                                        (amber_filter_df['Purity (bp)'] != 0) &
#                                        (amber_filter_df['Completeness (bp)'] != 0)
#                                        )]

amber_stack_df = amber_filter_df.set_index(['level', 'algorithm']).stack().reset_index()
amber_stack_df.columns = ['level', 'algorithm', 'statistic', 'score'
                            ]
amber_stack_df['level'] = ['domain' if x=='superkingdom' else x
                                for x in amber_stack_df['level']
                                ]
err_deduped_filter_df = err_deduped_df[['level', 'algorithm', 'statistic', 'score']]
err_deduped_filter_df = err_deduped_filter_df.loc[err_deduped_filter_df['statistic'].isin([
                                                    'precision', 'sensitivity']
                                                    )]
concat_tax_df = pd.concat([err_deduped_filter_df, amber_stack_df])
concat_tax_df.dropna(inplace=True)
concat_tax_df.to_csv('out.tsv', sep='\t')

cat_order = ['Kraken_0.10.5', 'Kraken_0.10.6', 'taxator-tk_1.3.0e', 'taxator-tk_1.4pre1e',
                'PhyloPythiaS_plus', 'SABer-Isolation Forest', 'SABer-OCSVM', 'SABer-GMM',
                'SABer-Ensemble'
                ]
g = sns.relplot(x='level', y='score', hue='statistic', style='statistic',
                col='algorithm', kind='line', col_wrap=3,
                col_order=cat_order,
                sort=False,
                data=concat_tax_df
                )
g.savefig("AMBER_SABer_relplot.png", bbox_inches='tight', dpi=300)
plt.close()
'''
sns.set(font_scale=1.5)  # crazy big
g = sns.relplot(x='level', y='score', hue='statistic', style='statistic',
                col='algorithm', kind='line', col_wrap=4,# ci="sd",
                col_order=['SABer-Isolation Forest', 'SABer-OCSVM', 'SABer-GMM',
                'SABer-Ensemble'],
                sort=False,
                data=err_deduped_df
                )
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_titles(row_template = '{row_name}', col_template = '{col_name}')

g.savefig(err_path + "SABer_relplot.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

g = sns.relplot(x='level', y='score', hue='statistic', style='statistic',
                col='algorithm', kind='line', col_wrap=5,# ci="sd",
                col_order=['MinHash', 'TPM', 'Isolation Forest', 'OCSVM', 'GMM',
                            'Tetra Ensemble', 'SABer-Isolation Forest',
                            'SABer-OCSVM', 'SABer-GMM', 'SABer-Ensemble'],
                sort=False,
                data=err_trim_df
                )
plt.ylim(0, 1)
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in g.axes.flat]
g.set_titles(row_template = '{row_name}', col_template = '{col_name}')

g.savefig(err_path + "SABer_AllSteps_relplot.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()

'''

# Stat by level line plot
g = sns.relplot(x='Taxonomic rank', y='score', hue='statistic', style='statistic',
                col='algorithm', kind='line', col_wrap=2, sort=False, data=amber_stack_df
                )
g.set_xticklabels(rotation=45)
g.savefig("AMBER_tax_relplot.pdf", bbox_inches='tight')

'''
'''
# Open SAG ID to AMBER map
s2a_map_file = 'SABer2AMBER_map.tsv'
s2a_map_df = pd.read_csv(s2a_map_file, header=0, sep='\t')
s2a_map_df['sag_id'] = [x.rsplit('.', 1)[0] for x in s2a_map_df['sag_id']]
deduped_df = unstack_df.loc[unstack_df['algorithm'].isin(['SABer-GMM', 'SABer-OCSVM',
                                                            'SABer-Isolation Forest',
                                                            'SABer-Ensemble'])
                                                            ]

deduped_df['sag_id'] = [x.replace('.mockSAG', '') for x in deduped_df['sag_id']]
deduped_amb_df = pd.merge(deduped_df, s2a_map_df, on='sag_id', how='left')
deduped_amb_df.columns = ['bin_id', 'algorithm', 'level', 'MCC', 'Precision',
                        'Sensitivity', 'genome_id'
                        ]
strain_df = deduped_amb_df.loc[deduped_amb_df['level'] == 'strain']
strain_filter_df = strain_df[['bin_id', 'algorithm', 'Precision', 'Sensitivity', 'genome_id']]

# AMBER Genome
amber_gen_file = 'AMBER_genome_results.tsv'
amber_gen_df = pd.read_csv(amber_gen_file, header=0, sep='\t')
amber_gen_df.columns = ['sample_id', 'bin_id', 'genome_id', 'Precision', 'Sensitivity',
                        'Predicted size (bp)', 'True positives (bp)', 'True size (bp)',
                        'Purity (seq)', 'Completeness (seq)', 'Predicted size (seq)',
                        'True positives (seq)', 'True size (seq)', 'algorithm'
                        ]
amber_filter_df = amber_gen_df[['bin_id', 'algorithm', 'Precision', 'Sensitivity', 'genome_id']]

sab_amb_df = pd.concat([strain_filter_df, amber_filter_df])

avg_sab_amb_df = sab_amb_df.groupby(['algorithm']).mean().reset_index()

paired_cmap_dict = {'blue': (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
                    'orange': (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
                    'green': (0.41568627450980394, 0.8, 0.39215686274509803),
                    'rose': (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
                    'purple': (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
                    'brown': (0.5490196078431373, 0.3803921568627451, 0.23529411764705882),
                    'pink': (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
                    'gray': (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
                    'gold': (0.8352941176470589, 0.7333333333333333, 0.403921568627451),
                    'lightblue': (0.5098039215686274, 0.7764705882352941, 0.8862745098039215)
                    }

# Plot average stats for algorithm
color_dict = {'Binsanity-wf_0.2.5.9': paired_cmap_dict['rose'],
                'Binsanity_0.2.5.9': paired_cmap_dict['rose'],
                'COCACOLA': paired_cmap_dict['purple'],
                'CONCOCT_2': paired_cmap_dict['lightblue'],
                'CONCOCT_CAMI': paired_cmap_dict['lightblue'],
                'DAS_Tool_1.1': paired_cmap_dict['blue'],
                'MaxBin_2.0.2_CAMI': paired_cmap_dict['gold'],
                'MaxBin_2.2.4': paired_cmap_dict['gold'],
                'MetaBAT_2.11.2': paired_cmap_dict['green'],
                'MetaBAT_CAMI': paired_cmap_dict['green'],
                'Metawatt_3.5_CAMI': paired_cmap_dict['brown'],
                'MyCC_CAMI': paired_cmap_dict['gray'],
                'SABer-Isolation Forest': paired_cmap_dict['orange'],
                'SABer-OCSVM': paired_cmap_dict['orange'],
                'SABer-GMM': paired_cmap_dict['orange'],
                'SABer-Ensemble': paired_cmap_dict['orange']
                }
marker_dict = {'MaxBin_2.0.2_CAMI': 'o', 'MetaBAT_CAMI': 'o',
                'MaxBin_2.2.4': 'd', 'CONCOCT_2': 'o', 'SABer-GMM': 'o',
                'Binsanity_0.2.5.9': 'o', 'Binsanity-wf_0.2.5.9': 'd',
                'SABer-Ensemble': 'd', 'COCACOLA': 'o', 'Metawatt_3.5_CAMI': 'o',
                'CONCOCT_CAMI': 'd', 'SABer-Isolation Forest': '^',
                'SABer-OCSVM': 's', 'MyCC_CAMI': 'o', 'DAS_Tool_1.1': 'd',
                'MetaBAT_2.11.2': 'd'
                }
cat_order = ['Binsanity-wf_0.2.5.9', 'Binsanity_0.2.5.9', 'COCACOLA', 'CONCOCT_2',
                'CONCOCT_CAMI', 'DAS_Tool_1.1', 'MaxBin_2.0.2_CAMI', 'MaxBin_2.2.4',
                'MetaBAT_2.11.2', 'MetaBAT_CAMI', 'Metawatt_3.5_CAMI', 'MyCC_CAMI',
                'SABer-Isolation Forest', 'SABer-OCSVM', 'SABer-GMM', 'SABer-Ensemble'
                ]

g = sns.scatterplot(x='Precision', y='Sensitivity', hue='algorithm', palette=color_dict,
                    style='algorithm', markers=marker_dict, hue_order=cat_order,
                    data=avg_sab_amb_df, s=75
                    )
g.set(xlabel='Average Precision', ylabel='Average Sensitivity')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend(edgecolor='b', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
g.figure.savefig("AMBER_SABer_scatterplot.png", bbox_inches='tight', dpi=300)

flierprops = dict(markerfacecolor='0.75', markersize=5, markeredgecolor='w',
              linestyle='none')

piv_sab_amb_df = sab_amb_df.set_index(['bin_id', 'algorithm', 'genome_id']).stack().reset_index()
piv_sab_amb_df.columns = ['bin_id', 'algorithm', 'genome_id', 'statistic', 'score']

cat_order = ['Binsanity-wf_0.2.5.9', 'Binsanity_0.2.5.9', 'COCACOLA', 'CONCOCT_2',
                'CONCOCT_CAMI', 'DAS_Tool_1.1', 'MaxBin_2.0.2_CAMI', 'MaxBin_2.2.4',
                'MetaBAT_2.11.2', 'MetaBAT_CAMI', 'Metawatt_3.5_CAMI', 'MyCC_CAMI',
                'SABer-Isolation Forest', 'SABer-OCSVM', 'SABer-GMM', 'SABer-Ensemble'
                ]
with sns.axes_style("white"):
    ax = sns.catplot(x="statistic", y="score", hue='algorithm', kind='box',
                        data=piv_sab_amb_df, aspect=2, palette=sns.light_palette("black"),
                        flierprops=flierprops, hue_order=cat_order)
    plt.plot([-1, 3], [0.25, 0.25], linestyle='--', alpha=0.3, color='k')
    plt.plot([-1, 3], [0.50, 0.50], linestyle='--', alpha=0.3, color='k')
    plt.plot([-1, 3], [0.75, 0.75], linestyle='--', alpha=0.3, color='k')

    plt.ylim(0, 1)
    plt.xlim(-0.5, 1.5)
    #plt.title('SAG-plus CAMI-1-High error analysis')
    ax._legend.set_title('Workflow\nStage')
    plt.savefig('AMBER_SABer_error_boxplox_count.png', bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()
'''
'''
# Stat by level line plot
for stat in set(err_df['statistic']):
    stat_err_df = err_df.loc[err_df['statistic'] == stat]
    g = sns.relplot(x='level', y='score', hue='statistic', style='statistic',
                    col='algorithm', kind='line', col_wrap=4,
                    col_order=['mockSAG', 'MinHash', 'TPM', 'GMM', 'OCSVM',
                                    'Isolation Forest', 'Tetra Ensemble', 'Final GMM',
                                    'Final OCSVM', 'Final Isolation Forest', 'Final Ensemble',
                                    'SABer-GMM', 'SABer-OCSVM', 'SABer-Isolation Forest',
                                    'SABer-Ensemble'
                                    ],
                    sort=False, ci='sd', n_boot=100,
                    data=stat_err_df
                    )
    g.savefig(stat + ".relplot.pdf", bbox_inches='tight')

'''
