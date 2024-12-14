import math
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from .observable_matrix import compute_probability_matrix
from .eval_motifs import getMotifMasks, calcMotifProbability, aggregateProbabilties_16, getDeviations
from lib.constants import *
from .motif_stats_gpu import MotifStatisticsGPU


class ConnectomeRealization:
    def __init__(self, df_summary, mode = "poisson", poisson_threshold=10):
        self.df_summary = df_summary
        self.mode = mode
        self.poisson_threshold = poisson_threshold  
    
    def generate(self, model_descriptor):
        assert model_descriptor in self.df_summary.columns  

        #num_pre_sites = self.df_summary["num_pre_sites"].values.astype(int)
        #num_post_sites = self.df_summary["num_post_sites"].values.astype(int)
        expected_synapse_count = self.df_summary[model_descriptor].values

        mask_integer = np.isclose(expected_synapse_count - np.round(expected_synapse_count), 0.0, atol=10**-6) \
        # mask_integer = (expected_synapse_count > self.poisson_threshold)  
        # mask_integer = np.zeros(expected_synapse_count.size, dtype=bool)
        mask_stochastic = ~mask_integer

        synapse_counts = np.zeros_like(expected_synapse_count)

        synapse_counts[mask_integer] = np.round(expected_synapse_count[mask_integer])
        
        if(self.mode == "poisson"):
            synapse_counts[mask_stochastic] = np.random.poisson(expected_synapse_count[mask_stochastic])
        elif(self.mode == "binomial"):
            num_synaptic_pairs = self.df_summary["num_synaptic_pairs"].values.astype(int)
            single_synapse_prob = expected_synapse_count / num_synaptic_pairs
            synapse_counts[mask_stochastic] = np.random.binomial(num_synaptic_pairs[mask_stochastic], single_synapse_prob[mask_stochastic])
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return synapse_counts.astype(int)


class Statistics():
    def __init__(self, index_data, num_selected_ids = None):
        self.index_data = index_data
        self.result = None
        self.parameters_by_model = {}

        self.mask_mapped_ids = (index_data.global_index.get_level_values("pre_id_mapped") >= 0) & (index_data.global_index.get_level_values("post_id_mapped") >= 0)
        self.masks, self.masks_inv = getMotifMasks(3)

        if(num_selected_ids is not None):
            self.motif_statistics_gpu = MotifStatisticsGPU(num_selected_ids)
        else:
            self.motif_statistics_gpu = MotifStatisticsGPU(index_data.num_ids_mapped)

    def delete_by_model(self, model_descriptor):
        self.result = self.result.drop(self.result[self.result.model_descriptor == model_descriptor].index)
        
    def get_aggregate_synapse_counts(self, df, column):
        by_celltype = df.groupby(["pre_celltype", "post_celltype"]).agg({column: "sum"})
        by_compartment = df.groupby(["post_compartment"]).agg({column: "sum"})
        by_celltype_compartment = df.groupby(["pre_celltype", "post_celltype", "post_compartment"]).agg({column: "sum"})
        return by_celltype, by_compartment, by_celltype_compartment  

    def get_cluster_distribution(self, syncounts, max_k, num_pairs, discrete):
        
        non_overlapping_pairs = num_pairs - syncounts.size

        if(discrete):
            k_np, counts_np = np.unique(syncounts, return_counts=True)
            k_np = k_np.astype(int).tolist()    
    
            bins = np.zeros(max_k+2)

            for idx, k in enumerate(k_np):
                if(k <= max_k):
                    bins[k] = counts_np[idx]
                else:
                    bins[max_k+1] += counts_np[idx]
                
            bins = np.array(bins)
            bins[0] += non_overlapping_pairs

            return np.arange(0, max_k+2).astype(int), np.array(bins).astype(int)
        
        else:
            lambd = syncounts
        
            n = lambd.shape[0]
            D = np.zeros((n, max_k+2))
            for k in range(0, max_k + 1):
                D[:, k] = (np.multiply(np.power(lambd, k), np.exp(-lambd)) / math.factorial(k)).flatten()
            
            D[:, max_k + 1] = 1 - np.sum(D, axis=1)
            bins = np.sum(D, axis=0)
            bins[0] += non_overlapping_pairs

            return np.arange(0, max_k+2).astype(int), bins 

    """
    def get_motif_distribution(self, synapse_matrix, discrete):
        n_rows, n_cols = synapse_matrix.shape
        assert n_rows == n_cols
        num_neurons = n_rows

        prob_matrix = compute_probability_matrix(synapse_matrix, discrete)

        num_triplet_samples = math.comb(num_neurons, 3)
        p_model = np.zeros((num_triplet_samples, 6))
        idx = 0
        for i in range(0, num_neurons):
            for j in range(i+1, num_neurons):
                for k in range(j+1, num_neurons):
                    p_model[idx, 0] = prob_matrix[i, j]
                    p_model[idx, 1] = prob_matrix[j, i]
                    p_model[idx, 2] = prob_matrix[i, k]
                    p_model[idx, 3] = prob_matrix[k, i]
                    p_model[idx, 4] = prob_matrix[j, k]
                    p_model[idx, 5] = prob_matrix[k, j]
                    idx += 1

        # time-consuming section 
        # --------------------------------------------------------
        p_average_all_pairs = np.mean(prob_matrix)

        p_model_inv = 1 - p_model
        p_avg = p_average_all_pairs * np.ones(6)
        p_avg = p_avg.reshape((-1, 6))
        p_avg_inv = 1 - p_avg
            
        motif_probabilities_64_random = {}
        motif_probabilities_64_model = {}
        for i in range(0, self.masks.shape[0]):
            mask = self.masks[i, :]
            mask_inv = self.masks_inv[i, :]
            motifKey = tuple(mask.astype(int))
            motif_probabilities_64_random[motifKey] = calcMotifProbability(p_avg, p_avg_inv, mask, mask_inv)
            motif_probabilities_64_model[motifKey] = calcMotifProbability(p_model, p_model_inv, mask, mask_inv)
        
        # --------------------------------------------------------

        motif_probabilities_16_random = aggregateProbabilties_16(motif_probabilities_64_random)
        motif_probabilities_16_model = aggregateProbabilties_16(motif_probabilities_64_model)
        
        deviations = getDeviations(motif_probabilities_16_random, motif_probabilities_16_model, idx_offset=1)
        return deviations
    """
  
    def compute(self, df, synapse_column, model_descriptor, realize=False, num_realizations=-1, parameter_index=None):        

        connectome_realization = ConnectomeRealization(df)
        
        def add_results_row(realization_index):
            new_row = pd.Series(index=self.columns)
            self.result = pd.concat([self.result, new_row.to_frame().T], axis=0, ignore_index=True)
            self.result.loc[self.result.index[-1], "model_descriptor"] = model_descriptor
            self.result.loc[self.result.index[-1], "parameter_index"] = realization_index
            self.result.parameter_index = self.result.parameter_index.astype(int)

        if(realize and (parameter_index is not None)):
            add_results_row(parameter_index)

            # generate realization of connectome
            synapses_realized = connectome_realization.generate(model_descriptor)
            df.loc[:, f"{model_descriptor}_realization"] = synapses_realized

            # determine cellular matrix between selected neurons
            matrix_cellular = np.zeros((self.index_data.num_ids_mapped, self.index_data.num_ids_mapped))
            syncounts_flat = []
            for mapped_pre_post_id, global_indices in self.index_data.neuron_indices.items():
                mapped_pre_id, mapped_post_id = mapped_pre_post_id
                syncount = synapses_realized[global_indices].sum()
                matrix_cellular[mapped_pre_id, mapped_post_id] = syncount
                syncounts_flat.append(syncount)
            
            self.matrix_cellular = matrix_cellular
            syncounts_flat = np.array(syncounts_flat)

            # compute statistics
            if(self.enabled_syncounts):
                self.compute_synapse_counts(df, f"{model_descriptor}_realization")

            if(self.enabled_clusters):
                clustersize, occurrences = self.get_cluster_distribution(syncounts_flat, self.max_k, self.index_data.num_neuron_pairs, True)
                for k in range(0, clustersize.size):
                    self.result.loc[self.result.index[-1], f"CLUSTER-{k}"] = occurrences[k]

            if(self.enabled_motifs):
                self.compute_motifs(matrix_cellular, True)

        elif(realize):
        
            assert num_realizations > 0
        
            for realization_idx in tqdm(range(num_realizations)):

                add_results_row(realization_idx)
                #print("realization ", realization_idx)
                #if(realization_idx % 50 == 0):
                #    print(realization_idx)

                synapses_realized = connectome_realization.generate(model_descriptor)
                df.loc[:, f"{model_descriptor}_realization"] = synapses_realized
                
                if(self.enabled_clusters or self.enabled_motifs):    
                    matrix_cellular = np.zeros((self.index_data.num_ids_mapped, self.index_data.num_ids_mapped))
                    syncounts_flat = []
                    for mapped_pre_post_id, global_indices in self.index_data.neuron_indices.items():
                        mapped_pre_id, mapped_post_id = mapped_pre_post_id
                        syncount = synapses_realized[global_indices].sum()
                        matrix_cellular[mapped_pre_id, mapped_post_id] = syncount
                        syncounts_flat.append(syncount)
                    
                    self.matrix_cellular = matrix_cellular
                    syncounts_flat = np.array(syncounts_flat)
                
                if(self.enabled_syncounts):
                    self.compute_synapse_counts(df, f"{model_descriptor}_realization")

                if(self.enabled_clusters):
                    clustersize, occurrences = self.get_cluster_distribution(syncounts_flat, self.max_k, self.index_data.num_neuron_pairs, True)
                    for k in range(0, clustersize.size):
                        self.result.loc[self.result.index[-1], f"CLUSTER-{k}"] = occurrences[k]

                if(self.enabled_motifs):
                    self.compute_motifs(matrix_cellular, True)

        else:
            df_synapses = df[synapse_column]
            
            matrix_cellular = np.zeros((self.index_data.num_ids_mapped, self.index_data.num_ids_mapped))
            syncounts_flat = []
            for mapped_pre_post_id, global_indices in self.index_data.neuron_indices.items():
                mapped_pre_id, mapped_post_id = mapped_pre_post_id
                syncount = df_synapses.iloc[global_indices].sum()
                matrix_cellular[mapped_pre_id, mapped_post_id] = syncount
                syncounts_flat.append(syncount)
            
            self.matrix_cellular = matrix_cellular
            syncounts_flat = np.array(syncounts_flat)

            assert (df[synapse_column] % 1 == 0).all() 
            add_results_row(-1)

            if(self.enabled_syncounts):
                self.compute_synapse_counts(df, synapse_column)

            if(self.enabled_clusters):
                clustersize, occurrences = self.get_cluster_distribution(syncounts_flat, self.max_k, self.index_data.num_neuron_pairs, True)
                for k in range(0, clustersize.size):
                    self.result.loc[self.result.index[-1], f"CLUSTER-{k}"] = occurrences[k]

            if(self.enabled_motifs):
                self.compute_motifs(matrix_cellular, True)
    
    def to_numpy(self, columns, model_descriptor):
        row_cond = self.result.model_descriptor == model_descriptor
        return self.result.loc[row_cond, columns].to_numpy()


