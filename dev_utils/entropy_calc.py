import glob
import os

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap

working_dir = '/home/ryan/Desktop/renyi_entropy/references/'
sample_list = glob.glob(os.path.join(working_dir, "*.tsv"))
############################################################################################
'''
# Calculate entropy for all references
entropy_list = []
for samp_file in sample_list:
    samp_id = samp_file.split('/')[-1].rsplit('.', 1)[0]
    if 'entropy' not in samp_id and 'table' not in samp_id:
        print(samp_id)
        if samp_id.rsplit('_', 1)[1][0].isdigit():
            samp_label = samp_id.rsplit('_', 1)[0]
            samp_rep = samp_id.rsplit('_', 1)[1]
        else:
            samp_label = samp_id
            samp_rep = 0
        cov_df = pd.read_csv(samp_file, sep='\t', header=0)
        cov_df['hash_id'] = [hashlib.sha256(x.encode(encoding='utf-8')).hexdigest()
                             for x in cov_df['contigName']
                             ]
        depth_sum = cov_df['totalAvgDepth'].sum()
        cov_df['relative_depth'] = [x/depth_sum for x in cov_df['totalAvgDepth']]
        cov_dist = dit.Distribution(cov_df['hash_id'].tolist(),
                                    cov_df['relative_depth'].tolist()
                                    )
        q_list = [0, 1, 2, 4, 8, 16, 32, np.inf]
        for q in q_list:
            r_ent = renyi_entropy(cov_dist, q)
            print(q, r_ent)
            entropy_list.append([samp_id, samp_label, samp_rep, q, r_ent])
ent_df = pd.DataFrame(entropy_list, columns=['sample_id', 'sample_type',
                                             'sample_rep', 'alpha',
                                             'Renyi_Entropy'
                                             ])
ent_df.to_csv(os.path.join(working_dir, 'entropy_table.tsv'), sep='\t', index=False)
'''
############################################################################################
# Build all the reference plots and run clustering
ent_df = pd.read_csv(os.path.join(working_dir, 'entropy_table.tsv'), sep='\t', header=0)
type_list = ent_df['sample_type'].unique()
rep_list = ent_df['sample_id'].unique()
samp2type = {x: y for x, y in zip(ent_df['sample_id'], ent_df['sample_type'])}
# Have to replace the np.inf with a real value for plotting
ent_df['alpha'].replace(4, 3, inplace=True)
ent_df['alpha'].replace(8, 4, inplace=True)
ent_df['alpha'].replace(16, 5, inplace=True)
ent_df['alpha'].replace(32, 6, inplace=True)
ent_df['alpha'].replace(np.inf, 7, inplace=True)
x_labels = {0: 'Richness (a=0)', 1: 'Shannon (a=1)',
            2: 'Simpson (a=2)', 3: '4', 4: '8', 5: '16',
            6: '32', 7: 'Berger–Parker (a=inf)'
            }
ent_df['x_labels'] = [x_labels[x] for x in ent_df['alpha']]
cpal = {x: y for x, y in zip(type_list, sns.color_palette(n_colors=len(type_list)))}
sns.set(rc={'figure.figsize': (12, 8)})
sns.set_style("white")
p = sns.catplot(x="x_labels", y="Renyi_Entropy", hue="sample_type", col="sample_type",
                data=ent_df, palette=cpal, col_wrap=3, legend_out=True
                )
p.map_dataframe(sns.boxplot, x="x_labels", y="Renyi_Entropy",
                data=ent_df, boxprops={'facecolor': 'None'},
                whiskerprops={'linewidth': 1},
                showfliers=False,  # showcaps=False,
                )
p.set_xticklabels(rotation=45)
lgd_dat = p._legend_data
p.add_legend(legend_data={x: lgd_dat[x] for x in lgd_dat.keys() if x in cpal.keys()})
p.savefig(os.path.join(working_dir, 'entropy_plot.png'), bbox_inches='tight')
plt.clf()
plt.close()

from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse

A = ent_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
A_sparse = sparse.csr_matrix(A)

similarities = euclidean_distances(A_sparse)
euc_df = pd.DataFrame(similarities, index=A.index.values, columns=A.index.values)
euc_unstack_df = euc_df.unstack().reset_index()
euc_unstack_df.columns = ['sample1', 'sample2', 'euclidean_distances']
euc_unstack_df['samp1_label'] = [samp2type[x] for x in euc_unstack_df['sample1']]
euc_unstack_df['samp2_label'] = [samp2type[x] for x in euc_unstack_df['sample2']]
sns.set(rc={'figure.figsize': (12, 8)})
sns.set_style("white")
b = sns.catplot(x="samp1_label", y="euclidean_distances", hue="samp2_label", col="samp2_label",
                data=euc_unstack_df, palette=cpal, col_wrap=3, legend_out=True
                )
b.map_dataframe(sns.boxplot, x="samp1_label", y="euclidean_distances",
                data=euc_unstack_df, boxprops={'facecolor': 'None'},
                whiskerprops={'linewidth': 1},
                showfliers=False,  # showcaps=False,
                )
b.set_xticklabels(rotation=45)
lgd_dat = b._legend_data
b.add_legend(legend_data={x: lgd_dat[x] for x in lgd_dat.keys() if x in cpal.keys()})
b.savefig(os.path.join(working_dir, 'similarity_plot.png'), bbox_inches='tight')
plt.clf()
plt.close()

umap_fit = umap.UMAP(n_neighbors=2, min_dist=0.0, n_components=2,
                     random_state=42
                     ).fit(A)
umap_emb = umap_fit.transform(A)
umap_df = pd.DataFrame(umap_emb, index=A.index.values)

clusterer = hdbscan.HDBSCAN(min_cluster_size=2, allow_single_cluster=True,
                            prediction_data=True
                            ).fit(umap_df)

cluster_labels = clusterer.labels_
cluster_probs = clusterer.probabilities_
cluster_outlier = clusterer.outlier_scores_

umap_df['sample_type'] = [samp2type[x] for x in umap_df.index.values]
umap_df['sample_id'] = umap_df.index.values
umap_df['cluster'] = cluster_labels
umap_df['probabilities'] = cluster_probs
umap_df['outlier_scores'] = cluster_outlier
samp2clust = {x: y for x, y in zip(umap_df['sample_id'], umap_df['cluster'])}

sns.set(rc={'figure.figsize': (12, 8)})
sns.set_style("white")
mark_list = ['.', 'v', '^', '<', '>', 's', 'P', 'D', 'p', '*']
b = sns.scatterplot(x=0, y=1, hue="sample_type", style='cluster',
                    data=umap_df, palette=cpal, markers=mark_list, s=200
                    )
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig = b.get_figure()
fig.savefig(os.path.join(working_dir, 'entropy_clusters.png'), bbox_inches='tight')
plt.clf()
plt.close()

ent_umap_df = ent_df.merge(umap_df, on=['sample_id', 'sample_type'], how='left')
sns.set(rc={'figure.figsize': (12, 8)})
sns.set_style("white")
p = sns.catplot(x="x_labels", y="Renyi_Entropy", hue="sample_type",
                col="cluster", data=ent_umap_df, col_wrap=3, legend=False
                )
p.map_dataframe(sns.boxplot, x="x_labels", y="Renyi_Entropy",
                data=ent_umap_df, boxprops={'facecolor': 'None'},
                whiskerprops={'linewidth': 1},
                showfliers=False
                )
p.set_xticklabels(rotation=45)
lgd_dat = p._legend_data
p.add_legend(legend_data={x: lgd_dat[x] for x in lgd_dat.keys() if x in cpal.keys()})
p.savefig(os.path.join(working_dir, 'clustent_plot.png'), bbox_inches='tight')
plt.clf()
plt.close()

# Closest ref sample methods
cmpr_list = []
for r1, row1 in A.iterrows():
    keep_diff = [r1, '', np.inf]
    for r2, row2 in A.iterrows():
        diff = ((row1 - row2).abs()).sum()
        if diff < keep_diff[2] and r1 != r2:
            keep_diff = [r1, r2, diff]
    cmpr_list.append(keep_diff)
cmpr_df = pd.DataFrame(cmpr_list, columns=['sample_id', 'best_match', 'diff'])
ent_best_df = ent_umap_df.merge(cmpr_df, on='sample_id', how='left')
ent_best_df.to_csv(os.path.join(working_dir, 'cluster_table.tsv'), sep='\t', index=False)

############################################################################################
# Cluster real samples
real_dir = '/home/ryan/Desktop/renyi_entropy/SI/'
real_list = glob.glob(os.path.join(real_dir, "*.tsv"))
'''
entropy_list = []
for samp_file in real_list:
    samp_id = samp_file.split('/')[-1].rsplit('.', 1)[0]
    if 'entropy' not in samp_id and 'table' not in samp_id:
        print(samp_id)
        if samp_id.rsplit('_', 1)[1][0].isdigit():
            samp_label = samp_id.rsplit('_', 1)[0]
            samp_rep = samp_id.rsplit('_', 1)[1]
        else:
            samp_label = samp_id
            samp_rep = 0
        cov_df = pd.read_csv(samp_file, sep='\t', header=0)
        cov_df['hash_id'] = [hashlib.sha256(x.encode(encoding='utf-8')).hexdigest()
                             for x in cov_df['contigName']
                             ]
        depth_sum = cov_df['totalAvgDepth'].sum()
        cov_df['relative_depth'] = [x/depth_sum for x in cov_df['totalAvgDepth']]
        cov_dist = dit.Distribution(cov_df['hash_id'].tolist(),
                                    cov_df['relative_depth'].tolist()
                                    )
        q_list = [0, 1, 2, 4, 8, 16, 32, np.inf]
        for q in q_list:
            r_ent = renyi_entropy(cov_dist, q)
            print(q, r_ent)
            entropy_list.append([samp_id, samp_label, samp_rep, q, r_ent])
real_df = pd.DataFrame(entropy_list, columns=['sample_id', 'sample_type',
                                             'sample_rep', 'alpha',
                                             'Renyi_Entropy'
                                             ])
real_df.to_csv(os.path.join(real_dir, 'entropy_table.tsv'), sep='\t', index=False)
'''
real_df = pd.read_csv(os.path.join(real_dir, 'entropy_table.tsv'), sep='\t', header=0)
samp2type = {x: y for x, y in zip(real_df['sample_id'], real_df['sample_type'])}

# Assign real data to clusters
real_A = real_df.pivot(index='sample_id', columns='alpha', values='Renyi_Entropy')
umap_emb = umap_fit.transform(real_A)
umap_df = pd.DataFrame(umap_emb, index=real_A.index.values)

test_labels, strengths = hdbscan.approximate_predict(clusterer, umap_df)
cluster_labels = test_labels
cluster_probs = strengths
umap_df['sample_type'] = [samp2type[x] for x in umap_df.index.values]
umap_df['sample_id'] = umap_df.index.values
umap_df['cluster'] = cluster_labels
umap_df['probabilities'] = cluster_probs

# Closest ref sample methods
r_cmpr_list = []
for r1, row1 in real_A.iterrows():
    keep_diff = [r1, '', np.inf]
    for r2, row2 in A.iterrows():
        diff = ((row1 - row2).abs()).sum()
        if diff < keep_diff[2] and r1 != r2:
            keep_diff = [r1, r2, diff]
    r_cmpr_list.append(keep_diff)
r_cmpr_df = pd.DataFrame(r_cmpr_list, columns=['sample_id', 'best_match', 'diff'])
best_df = umap_df.merge(r_cmpr_df, on='sample_id', how='left')
best_df.to_csv(os.path.join(real_dir, 'cluster_table.tsv'), sep='\t', index=False)
