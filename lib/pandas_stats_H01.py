
from .pandas_stats_impl import Statistics 
import pandas as pd
import numpy as np

import lib.H01_preprocessing.h01_constants as H01

class H01AggregateStatistics(Statistics):
    def __init__(self, index_data, compute_syncounts = True, compute_clusters = True, compute_motifs = True, max_k=19):
        self.enabled_syncounts = compute_syncounts
        self.enabled_clusters = compute_clusters
        self.enabled_motifs = compute_motifs
        self.max_k = max_k


        super().__init__(index_data)

        self.columns = ["model_descriptor", "parameter_index",
           "ALL_ALL", 

           "EXC_EXC", "EXC_INH", "EXC_OTHER", "EXC_UNKNOWN", 
           "INH_EXC", "INH_INH", "INH_OTHER", "INH_UNKNOWN", 
           "OTHER_EXC", "OTHER_INH", "OTHER_OTHER", "OTHER_UNKNOWN",
           "UNKNOWN_EXC", "UNKNOWN_INH", "UNKNOWN_OTHER", "UNKNOWN_UNKNOWN",

           "EXC_EXC-SOMA", "EXC_EXC-DEND", "EXC_EXC-AIS", 
           "EXC_INH-SOMA", "EXC_INH-DEND", "EXC_INH-AIS",
           "EXC_OTHER-SOMA", "EXC_OTHER-DEND", "EXC_OTHER-AIS",
           "EXC_UNKNOWN-SOMA", "EXC_UNKNOWN-DEND", "EXC_UNKNOWN-AIS",

           "INH_EXC-SOMA", "INH_EXC-DEND", "INH_EXC-AIS", 
           "INH_INH-SOMA", "INH_INH-DEND", "INH_INH-AIS",
           "INH_OTHER-SOMA", "INH_OTHER-DEND", "INH_OTHER-AIS",
           "INH_UNKNOWN-SOMA", "INH_UNKNOWN-DEND", "INH_UNKNOWN-AIS",
           
           "OTHER_EXC-SOMA", "OTHER_EXC-DEND", "OTHER_EXC-AIS", 
           "OTHER_INH-SOMA", "OTHER_INH-DEND", "OTHER_INH-AIS",
           "OTHER_OTHER-SOMA", "OTHER_OTHER-DEND", "OTHER_OTHER-AIS",
           "OTHER_UNKNOWN-SOMA", "OTHER_UNKNOWN-DEND", "OTHER_UNKNOWN-AIS",
           
           "UNKNOWN_EXC-SOMA", "UNKNOWN_EXC-DEND", "UNKNOWN_EXC-AIS", 
           "UNKNOWN_INH-SOMA", "UNKNOWN_INH-DEND", "UNKNOWN_INH-AIS",
           "UNKNOWN_OTHER-SOMA", "UNKNOWN_OTHER-DEND", "UNKNOWN_OTHER-AIS",
           "UNKNOWN_UNKNOWN-SOMA", "UNKNOWN_UNKNOWN-DEND", "UNKNOWN_UNKNOWN-AIS",
           
           "CLUSTER-0", "CLUSTER-1", "CLUSTER-2", "CLUSTER-3", "CLUSTER-4", "CLUSTER-5", "CLUSTER-6", "CLUSTER-7", "CLUSTER-8", "CLUSTER-9", "CLUSTER-10",
           "CLUSTER-11", "CLUSTER-12", "CLUSTER-13", "CLUSTER-14", "CLUSTER-15", "CLUSTER-16", "CLUSTER-17", "CLUSTER-18", "CLUSTER-19", "CLUSTER-20",
           
           "MOTIF-1", "MOTIF-2", "MOTIF-3", "MOTIF-4", "MOTIF-5", "MOTIF-6", "MOTIF-7", "MOTIF-8",
           "MOTIF-9", "MOTIF-10", "MOTIF-11", "MOTIF-12", "MOTIF-13", "MOTIF-14", "MOTIF-15", "MOTIF-16"]

        self.dtypes = {
            "model_descriptor": "object",
            "parameter_index": "int64",
        }
        for col in self.columns[2:]:
            self.dtypes[col] = "float64"
           
        self.result = pd.DataFrame(columns=self.columns).astype(self.dtypes)



    def compute_clusters(self, syncounts, num_pairs, max_k, discrete):
        clustersize, occurrences = self.get_cluster_distribution(syncounts, max_k, num_pairs, discrete)
        for k in range(0, clustersize.size):
            self.result.loc[self.result.index[-1], f"CLUSTER-{k}"] = occurrences[k]
        

    def compute_synapse_counts(self, df, synapse_column):
        by_celltype, by_compartment, by_celltype_compartment = self.get_aggregate_synapse_counts(df, synapse_column)

        def count_by_ct(pre_ct, post_ct):
            condition = (by_celltype.index.get_level_values("pre_celltype").isin(pre_ct)) & (by_celltype.index.get_level_values("post_celltype").isin(post_ct))
            return by_celltype[condition][synapse_column].sum()
        
        def count_by_ct_compartment(pre_ct, post_ct, post_compartment):
            condition = (by_celltype_compartment.index.get_level_values("pre_celltype").isin(pre_ct)) & (by_celltype_compartment.index.get_level_values("post_celltype").isin(post_ct)) \
                & (by_celltype_compartment.index.get_level_values("post_compartment").isin(post_compartment))
            return by_celltype_compartment[condition][synapse_column].sum()

        
        self.result.loc[self.result.index[-1], "ALL_ALL"] = by_celltype[synapse_column].sum()
        
        self.result.loc[self.result.index[-1], "EXC_EXC"] = count_by_ct(H01.EXC, H01.EXC)
        self.result.loc[self.result.index[-1], "EXC_INH"] = count_by_ct(H01.EXC, H01.INH)
        self.result.loc[self.result.index[-1], "EXC_OTHER"] = count_by_ct(H01.EXC, H01.OTHER)
        self.result.loc[self.result.index[-1], "EXC_UNKNOWN"] = count_by_ct(H01.EXC, H01.UNKNOWN)

        self.result.loc[self.result.index[-1], "INH_EXC"] = count_by_ct(H01.INH, H01.EXC)
        self.result.loc[self.result.index[-1], "INH_INH"] = count_by_ct(H01.INH, H01.INH)
        self.result.loc[self.result.index[-1], "INH_OTHER"] = count_by_ct(H01.INH, H01.OTHER)
        self.result.loc[self.result.index[-1], "INH_UNKNOWN"] = count_by_ct(H01.INH, H01.UNKNOWN)

        self.result.loc[self.result.index[-1], "OTHER_EXC"] = count_by_ct(H01.OTHER, H01.EXC)
        self.result.loc[self.result.index[-1], "OTHER_INH"] = count_by_ct(H01.OTHER, H01.INH)
        self.result.loc[self.result.index[-1], "OTHER_OTHER"] = count_by_ct(H01.OTHER, H01.OTHER)
        self.result.loc[self.result.index[-1], "OTHER_UNKNOWN"] = count_by_ct(H01.OTHER, H01.UNKNOWN)

        self.result.loc[self.result.index[-1], "UNKNOWN_EXC"] = count_by_ct(H01.UNKNOWN, H01.EXC)
        self.result.loc[self.result.index[-1], "UNKNOWN_INH"] = count_by_ct(H01.UNKNOWN, H01.INH)
        self.result.loc[self.result.index[-1], "UNKNOWN_OTHER"] = count_by_ct(H01.UNKNOWN, H01.OTHER)
        self.result.loc[self.result.index[-1], "UNKNOWN_UNKNOWN"] = count_by_ct(H01.UNKNOWN, H01.UNKNOWN)
        
        self.result.loc[self.result.index[-1], "EXC_EXC-SOMA"] = count_by_ct_compartment(H01.EXC, H01.EXC, H01.SOMA)
        self.result.loc[self.result.index[-1], "EXC_EXC-DEND"] = count_by_ct_compartment(H01.EXC, H01.EXC, H01.DEND)
        self.result.loc[self.result.index[-1], "EXC_EXC-AIS"] = count_by_ct_compartment(H01.EXC, H01.EXC, H01.AIS)
        self.result.loc[self.result.index[-1], "EXC_INH-SOMA"] = count_by_ct_compartment(H01.EXC, H01.INH, H01.SOMA)
        self.result.loc[self.result.index[-1], "EXC_INH-DEND"] = count_by_ct_compartment(H01.EXC, H01.INH, H01.DEND)
        self.result.loc[self.result.index[-1], "EXC_INH-AIS"] = count_by_ct_compartment(H01.EXC, H01.INH, H01.AIS)
        self.result.loc[self.result.index[-1], "EXC_OTHER-SOMA"] = count_by_ct_compartment(H01.EXC, H01.OTHER, H01.SOMA)
        self.result.loc[self.result.index[-1], "EXC_OTHER-DEND"] = count_by_ct_compartment(H01.EXC, H01.OTHER, H01.DEND)
        self.result.loc[self.result.index[-1], "EXC_OTHER-AIS"] = count_by_ct_compartment(H01.EXC, H01.OTHER, H01.AIS)
        self.result.loc[self.result.index[-1], "EXC_UNKNOWN-SOMA"] = count_by_ct_compartment(H01.EXC, H01.UNKNOWN, H01.SOMA)
        self.result.loc[self.result.index[-1], "EXC_UNKNOWN-DEND"] = count_by_ct_compartment(H01.EXC, H01.UNKNOWN, H01.DEND)
        self.result.loc[self.result.index[-1], "EXC_UNKNOWN-AIS"] = count_by_ct_compartment(H01.EXC, H01.UNKNOWN, H01.AIS)

        self.result.loc[self.result.index[-1], "INH_EXC-SOMA"] = count_by_ct_compartment(H01.INH, H01.EXC, H01.SOMA)
        self.result.loc[self.result.index[-1], "INH_EXC-DEND"] = count_by_ct_compartment(H01.INH, H01.EXC, H01.DEND)
        self.result.loc[self.result.index[-1], "INH_EXC-AIS"] = count_by_ct_compartment(H01.INH, H01.EXC, H01.AIS)
        self.result.loc[self.result.index[-1], "INH_INH-SOMA"] = count_by_ct_compartment(H01.INH, H01.INH, H01.SOMA)
        self.result.loc[self.result.index[-1], "INH_INH-DEND"] = count_by_ct_compartment(H01.INH, H01.INH, H01.DEND)
        self.result.loc[self.result.index[-1], "INH_INH-AIS"] = count_by_ct_compartment(H01.INH, H01.INH, H01.AIS)
        self.result.loc[self.result.index[-1], "INH_OTHER-SOMA"] = count_by_ct_compartment(H01.INH, H01.OTHER, H01.SOMA)
        self.result.loc[self.result.index[-1], "INH_OTHER-DEND"] = count_by_ct_compartment(H01.INH, H01.OTHER, H01.DEND)
        self.result.loc[self.result.index[-1], "INH_OTHER-AIS"] = count_by_ct_compartment(H01.INH, H01.OTHER, H01.AIS)
        self.result.loc[self.result.index[-1], "INH_UNKNOWN-SOMA"] = count_by_ct_compartment(H01.INH, H01.UNKNOWN, H01.SOMA)
        self.result.loc[self.result.index[-1], "INH_UNKNOWN-DEND"] = count_by_ct_compartment(H01.INH, H01.UNKNOWN, H01.DEND)
        self.result.loc[self.result.index[-1], "INH_UNKNOWN-AIS"] = count_by_ct_compartment(H01.INH, H01.UNKNOWN, H01.AIS)

        self.result.loc[self.result.index[-1], "OTHER_EXC-SOMA"] = count_by_ct_compartment(H01.OTHER, H01.EXC, H01.SOMA)
        self.result.loc[self.result.index[-1], "OTHER_EXC-DEND"] = count_by_ct_compartment(H01.OTHER, H01.EXC, H01.DEND)
        self.result.loc[self.result.index[-1], "OTHER_EXC-AIS"] = count_by_ct_compartment(H01.OTHER, H01.EXC, H01.AIS)
        self.result.loc[self.result.index[-1], "OTHER_INH-SOMA"] = count_by_ct_compartment(H01.OTHER, H01.INH, H01.SOMA)
        self.result.loc[self.result.index[-1], "OTHER_INH-DEND"] = count_by_ct_compartment(H01.OTHER, H01.INH, H01.DEND)
        self.result.loc[self.result.index[-1], "OTHER_INH-AIS"] = count_by_ct_compartment(H01.OTHER, H01.INH, H01.AIS)
        self.result.loc[self.result.index[-1], "OTHER_OTHER-SOMA"] = count_by_ct_compartment(H01.OTHER, H01.OTHER, H01.SOMA)
        self.result.loc[self.result.index[-1], "OTHER_OTHER-DEND"] = count_by_ct_compartment(H01.OTHER, H01.OTHER, H01.DEND)
        self.result.loc[self.result.index[-1], "OTHER_OTHER-AIS"] = count_by_ct_compartment(H01.OTHER, H01.OTHER, H01.AIS)
        self.result.loc[self.result.index[-1], "OTHER_UNKNOWN-SOMA"] = count_by_ct_compartment(H01.OTHER, H01.UNKNOWN, H01.SOMA)
        self.result.loc[self.result.index[-1], "OTHER_UNKNOWN-DEND"] = count_by_ct_compartment(H01.OTHER, H01.UNKNOWN, H01.DEND)
        self.result.loc[self.result.index[-1], "OTHER_UNKNOWN-AIS"] = count_by_ct_compartment(H01.OTHER, H01.UNKNOWN, H01.AIS)

        self.result.loc[self.result.index[-1], "UNKNOWN_EXC-SOMA"] = count_by_ct_compartment(H01.UNKNOWN, H01.EXC, H01.SOMA)
        self.result.loc[self.result.index[-1], "UNKNOWN_EXC-DEND"] = count_by_ct_compartment(H01.UNKNOWN, H01.EXC, H01.DEND)
        self.result.loc[self.result.index[-1], "UNKNOWN_EXC-AIS"] = count_by_ct_compartment(H01.UNKNOWN, H01.EXC, H01.AIS)
        self.result.loc[self.result.index[-1], "UNKNOWN_INH-SOMA"] = count_by_ct_compartment(H01.UNKNOWN, H01.INH, H01.SOMA)
        self.result.loc[self.result.index[-1], "UNKNOWN_INH-DEND"] = count_by_ct_compartment(H01.UNKNOWN, H01.INH, H01.DEND)
        self.result.loc[self.result.index[-1], "UNKNOWN_INH-AIS"] = count_by_ct_compartment(H01.UNKNOWN, H01.INH, H01.AIS)
        self.result.loc[self.result.index[-1], "UNKNOWN_OTHER-SOMA"] = count_by_ct_compartment(H01.UNKNOWN, H01.OTHER, H01.SOMA)
        self.result.loc[self.result.index[-1], "UNKNOWN_OTHER-DEND"] = count_by_ct_compartment(H01.UNKNOWN, H01.OTHER, H01.DEND)
        self.result.loc[self.result.index[-1], "UNKNOWN_OTHER-AIS"] = count_by_ct_compartment(H01.UNKNOWN, H01.OTHER, H01.AIS)
        self.result.loc[self.result.index[-1], "UNKNOWN_UNKNOWN-SOMA"] = count_by_ct_compartment(H01.UNKNOWN, H01.UNKNOWN, H01.SOMA)
        self.result.loc[self.result.index[-1], "UNKNOWN_UNKNOWN-DEND"] = count_by_ct_compartment(H01.UNKNOWN, H01.UNKNOWN, H01.DEND)
        self.result.loc[self.result.index[-1], "UNKNOWN_UNKNOWN-AIS"] = count_by_ct_compartment(H01.UNKNOWN, H01.UNKNOWN, H01.AIS)

    def compute_motifs(self, syncount_matrix, discrete):
        #deviations = self.get_motif_distribution(syncount_matrix, discrete)
        deviations = self.motif_statistics_gpu.get_motif_distribution(syncount_matrix, discrete)
        for k in range(0, 16):
            self.result.loc[self.result.index[-1], f"MOTIF-{k+1}"] = deviations[k]