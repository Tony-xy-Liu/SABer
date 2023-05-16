import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import umap
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages



tax_file = "/home/ryan/SABer_local/SI/QC_run/SAGs/dedupe/dedupe_mqhp.tsv"
tax_df = pd.read_csv(tax_file, sep='\t', header=0)
xpg_file = "/home/ryan/SABer_local/SI/QC_run/SI_xPGs/dedupe/dedupe_mqhp.tsv"
xpg_df = pd.read_csv(xpg_file, sep='\t', header=0)
xpg_list = list(xpg_df['SAG_ID'].unique())
tax_df = tax_df.query("Genome_Id in @xpg_list")
print(tax_df.head())

tax_df['domain'] = [x.split(';')[0].split('__')[1] if "d__" in x
					else "Unclassified" for x in tax_df['classification']
					]
tax_df['family'] = [x.split(';')[4].split('__')[1] if ";f__" in x
					else "Unclassified" for x in tax_df['classification']
					]
tax_df['genus'] = [x.split(';')[5].split('__')[1] if ";g__" in x
					else "Unclassified" for x in tax_df['classification']
					]
#tax_df = tax_df.query("family == 'Thioglobaceae'")
tax_df = tax_df.query("domain == 'Bacteria'")
sag2fam = {x:y for x,y in zip(tax_df['Genome_Id'], tax_df['family'])}
sag2gen = {x:y for x,y in zip(tax_df['Genome_Id'], tax_df['genus'])}


emb_file = '/home/ryan/SABer_local/SI/QC_run/xPGs/SI060_150m/SABer_output/majority_rule/very_strict/SI060_150m.2k.merged_emb.tsv'
emb_df = pd.read_csv(emb_file, sep='\t', header=0).set_index('subcontig_id')

#emb_df['contig_id'] = [x.rsplit('_', 1)[0] for x in emb_df.index]
#grp_df = emb_df.groupby(['contig_id'])['0_cov', '1_cov', '0_tetra', '1_tetra'].mean().reset_index().set_index('contig_id')
#print(grp_df.head())

fav_list = ['AB-751_B23_AB-904', 'AB-750_M18_AB-904']
mhr_labs = "/home/ryan/SABer_local/SI/QC_run/xPGs/SI060_150m/SABer_output/SI060_150m.2k.201.mhr_contig_recruits.tsv"
mhr_df = pd.read_csv(mhr_labs, sep='\t', header=0)
#mhr_df = mhr_df.query("sag_id in @fav_list & jacc_sim == 1.0")[
mhr_df = mhr_df.query("jacc_sim == 1.0")[
					  ['sag_id', 'q_contig_id']].drop_duplicates()
mhr2sag = {x:y for x,y in zip(mhr_df['q_contig_id'], mhr_df['sag_id'])}

clust_labs = "/home/ryan/SABer_local/SI/QC_run/xPGs/SI060_150m/SABer_output/majority_rule/very_strict/SI060_150m.2k.inter_clusters.tsv"
clust_df = pd.read_csv(clust_labs, sep='\t', header=0)
#clust_df = clust_df.query("best_label in @fav_list")[
clust_df = clust_df[['best_label', 'contig_id']].drop_duplicates()
clust2sag = {x:y for x,y in zip(clust_df['contig_id'], clust_df['best_label'])}

mhr_tax_df = mhr_df.merge(tax_df[['Genome_Id', 'family', 'genus']],
						  left_on='sag_id', right_on='Genome_Id'
						  ).groupby(['q_contig_id', 'genus'])['sag_id'].size().reset_index()

best_anchors = list(mhr_tax_df['q_contig_id'].unique()) # .query("sag_id > 1")['q_contig_id'].unique())

clust_tax_df = clust_df.merge(tax_df[['Genome_Id', 'family', 'genus']],
						  left_on='best_label', right_on='Genome_Id'
						  ).groupby(['contig_id', 'genus'])['best_label'].size().reset_index()
best_clusters = list(clust_tax_df['contig_id'].unique()) # .query("best_label > 1")['contig_id'].unique())

mhr2int = {}
c = 0
for bl in list(mhr_df['sag_id'].unique()):
	mhr2int[bl] = c
	c+=1
gen2int = {}
d = 0
for g in sag2fam.values():
	if g not in gen2int.keys():
		gen2int[g] = d
		d+=1
mhr_dict = {x:mhr2int[y] for x,y in zip(mhr_df['q_contig_id'], mhr_df['sag_id'])
			if x in best_anchors
			}
#sag2gen_dict = {x:gen2int[sag2fam[y]] for x,y in zip(mhr_df['q_contig_id'], mhr_df['sag_id'])
#				if x in best_anchors
#				}
y_labels = [mhr_dict[x.rsplit('_', 1)[0]] if x.rsplit('_', 1)[0]
			in list(mhr_dict.keys()) else np.nan for x in emb_df.index
			]
emb_df['y_labels'] = y_labels
anchor_df = emb_df.query("~y_labels.isna()")
anchor_labels = list(anchor_df['y_labels'])
anchor_df.drop(columns=['y_labels'], inplace=True)
emb_df.drop(columns=['y_labels'], inplace=True)
reducer = umap.UMAP(min_dist=0.1, n_neighbors=15, metric='manhattan', 
					n_components=2, random_state=42
					)
umap_fit = reducer.fit(anchor_df.values, anchor_labels)
umap_emb = umap_fit.transform(emb_df.values)
#umap_emb = reducer.fit_transform(emb_df.values)
umap_df = pd.DataFrame(umap_emb, index=emb_df.index).reset_index()
umap_df['contig_id'] = [x.rsplit('_', 1)[0] for x in umap_df['subcontig_id']]

'''
recruited_list = []
for fav in fav_list:
	sub_clust_df = clust_df.query("best_label == @fav")
	sub_umap_df = umap_df.merge(sub_clust_df, on='contig_id')
	sub_umap_df['type'] = sub_umap_df['best_label']
	sub_umap_df['stage'] = ['anchor' if x in list(mhr_df['q_contig_id'])
							 else 'cluster' for x in sub_umap_df['contig_id']
							 ]
	recruited_list.append(sub_umap_df)
recruited_df = pd.concat(recruited_list)
'''
recruited_df = umap_df.merge(clust_df, on='contig_id')
recruited_df['type'] = 'recruited'
recruited_df['stage'] = ['anchor' if x in best_anchors
						 else 'not_recruited' for x in recruited_df['contig_id']
						 ]
recruited_df['stage'] = ['cluster' if x in best_clusters
						 else y for x,y in 
						 zip(recruited_df['contig_id'], recruited_df['stage'])
						 ]

recruit_list = list(recruited_df['contig_id'].unique())
unrecruited_df = umap_df.query("contig_id not in @recruit_list")
unrecruited_df['type'] = 'not_recruited'
unrecruited_df['stage'] = 'not_recruited'

cat_df = pd.concat([recruited_df[['subcontig_id', 0, 1, 'type', 'stage']],
					unrecruited_df[['subcontig_id', 0, 1, 'type', 'stage']]]
					)
genome_id = []
for sc in list(cat_df['subcontig_id']):
	c_id = sc.rsplit('_', 1)[0]
	if c_id in list(mhr2sag.keys()):
		g_id = mhr2sag[c_id]
	elif c_id in list(clust2sag.keys()):
		g_id = clust2sag[c_id]
	else:
		g_id = None

	genome_id.append(g_id)
cat_df['Genome_Id'] = genome_id
cat_df['family'] = [sag2fam[s] if s in list(sag2fam.keys())
					else None for s in cat_df['Genome_Id']
					]
cat_df['genus'] = [sag2gen[s] if s in list(sag2fam.keys())
					else None for s in cat_df['Genome_Id']
					]
'''
sctr = sns.scatterplot(data=cat_df.query("type == 'not_recruited'"),
					   x=0, y=1, hue='type', palette=['gray'],
					   alpha=0.50, marker='.', linewidth=0.1, s=10,
					   legend=False
					   )
sctr = sns.scatterplot(data=cat_df.query("type != 'not_recruited'"),
					   x=0, y=1, hue='family', alpha=0.50,
					   linewidth=0.1, markers='o', s=10,
					   )
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#plt.legend('', frameon=False)

sctr.figure.savefig("/home/ryan/SABer_local/SI/QC_run/xPGs/family_cluster_umap.pdf", dpi=300, bbox_inches='tight')
plt.clf()
plt.close()
'''
family_cmap = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
genus_cmap = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
family_list = list(cat_df['family'].unique())
genus_list = list(cat_df['genus'].unique())
with PdfPages("/home/ryan/SABer_local/SI/QC_run/xPGs/family_cluster_umap.pdf") as pdf_pages:
	for i,f_id in enumerate(family_list):
		if f_id != None:
			print(f_id)
			f_pal = family_cmap[i]
			fam_cat_df = cat_df.query("family == @f_id")
			non_cat_df = cat_df.query("family != @f_id | type == 'not_recruited'")
			non_cat_df['type'] = 'not_recruited'
			print(fam_cat_df.head())
			print(non_cat_df.head())
			figu = plt.figure()
			sctr = sns.scatterplot(data=non_cat_df, x=0, y=1, hue='type', palette=['gray'],
								   alpha=0.50, marker='.', linewidth=0.1, s=10,
								   legend=False
								   )
			sctr = sns.scatterplot(data=fam_cat_df, x=0, y=1, hue='family', palette=[f_pal],
								   alpha=0.50, linewidth=0.1, markers='o', s=10,
								   legend=False, 
								   ).set_title(f_id)
			#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
			#plt.legend('', frameon=False)

			pdf_pages.savefig(figu)
			plt.clf()
			plt.close()

with PdfPages("/home/ryan/SABer_local/SI/QC_run/xPGs/genus_cluster_umap.pdf") as pdf_pages:
	for i,g_id in enumerate(genus_list):
		if g_id != None:
			print(f_id)
			g_pal = genus_cmap[i]
			gen_cat_df = cat_df.query("genus == @g_id")
			non_cat_df = cat_df.query("genus != @g_id | type == 'not_recruited'")
			non_cat_df['type'] = 'not_recruited'
			print(gen_cat_df.head())
			print(non_cat_df.head())
			figu = plt.figure()
			sctr = sns.scatterplot(data=non_cat_df, x=0, y=1, hue='type', palette=['gray'],
								   alpha=0.50, marker='.', linewidth=0.1, s=10,
								   legend=False
								   )
			sctr = sns.scatterplot(data=gen_cat_df, x=0, y=1, hue='genus', palette=[g_pal],
								   alpha=0.50, linewidth=0.1, markers='o', s=10,
								   legend=False, 
								   ).set_title(g_id)
			#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
			#plt.legend('', frameon=False)

			pdf_pages.savefig(figu)
			plt.clf()
			plt.close()