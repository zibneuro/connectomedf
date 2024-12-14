import numpy as np
import pandas as pd
import pickle

from lib.constants import *

def get_local_pre_features(synapses_overlap_volume):
    pre_columns = [col_name for col_name in synapses_overlap_volume.index.names if "pre" in col_name]
    df_pre = synapses_overlap_volume.groupby(pre_columns).agg({"synapse_count": "sum"})
    df_pre.rename(columns={"synapse_count": "pre_contact_sites"}, inplace=True)
    return df_pre

def get_local_post_features(synapses_overlap_volume):
    post_columns = [col_name for col_name in synapses_overlap_volume.index.names if "post" in col_name]
    df_post = synapses_overlap_volume.groupby(post_columns).agg({"synapse_count": "sum"})
    df_post.rename(columns={"synapse_count": "post_contact_sites"}, inplace=True)
    return df_post

"""
import jax
import jax.numpy as jnp

from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
"""

def correct_self_connections(overlap_volume):
    pre_id = overlap_volume.index.get_level_values("pre_id_mapped")
    post_id = overlap_volume.index.get_level_values("post_id_mapped")
    cond_self_connection_flat = (pre_id == post_id) & (pre_id != -1)
    self_connections = overlap_volume.index[cond_self_connection_flat].values
    if(self_connections.size):
        n_pre = overlap_volume.pre_features.shape[0]
        n_post = overlap_volume.post_features.shape[0]

        M_null_flat = overlap_volume.M_null.copy()
        M_null = M_null_flat.reshape((n_pre, n_post))
        
        M_synaptic_pairs_flat = overlap_volume.M_synaptic_pairs_flat.copy()
        num_pre_sites = overlap_volume.num_pre_sites.copy()
        num_post_sites = overlap_volume.num_post_sites.copy()
        
        cond_self_connection = cond_self_connection_flat.reshape((n_pre, n_post))

        sum_rows_ref = M_null.sum(axis=1)
        M_null[cond_self_connection] = 0

        sum_rows = M_null.sum(axis=1)
        correction_factors = np.divide(sum_rows_ref, sum_rows)
        M_null_corrected = M_null * correction_factors[:, np.newaxis]
        M_null_corrected_flat = M_null_corrected.flatten()[~cond_self_connection_flat]

        M_synaptic_pairs_corrected_flat = M_synaptic_pairs_flat[~cond_self_connection_flat]
        num_pre_sites_corrected = num_pre_sites[~cond_self_connection_flat]
        num_post_sites_corrected = num_post_sites[~cond_self_connection_flat]

        index_corrected = overlap_volume.index.drop(self_connections)

        return index_corrected, M_null_corrected_flat, M_null_corrected, M_synaptic_pairs_corrected_flat, num_pre_sites_corrected, num_post_sites_corrected
    else:
        return overlap_volume.index, overlap_volume.M_null_flat, overlap_volume.M_null, overlap_volume.M_synaptic_pairs_flat, overlap_volume.num_pre_sites, overlap_volume.num_post_sites


class OverlapVolume:
    def __init__(self, overlap_volume_id, pre_features, post_features, global_index_names):
        self.overlap_volume_id = overlap_volume_id
        
        self.pre_features = pre_features
        self.post_features = post_features
        self.num_synapses = pre_features.pre_contact_sites.sum()

        self.init_product_index(global_index_names)
        self.init_compute_data()

    def init_product_index(self, global_index_names):
        pre_index = self.pre_features.index
        post_index = self.post_features.index
        
        global_pos_pre = [global_index_names.index(name) for name in pre_index.names]
        global_pos_post = [global_index_names.index(name) for name in post_index.names]
        if("overlap_volume" in global_index_names):
            global_pos_overlap = global_index_names.index("overlap_volume")
        else:
            global_pos_overlap = None
        
        product_indices_np = np.full((len(pre_index) * len(post_index), len(global_index_names)), np.nan)
        
        k = 0
        for pre_tuple in pre_index:
            for post_tuple in post_index:

                product_indices_np[k, global_pos_pre] = pre_tuple
                product_indices_np[k, global_pos_post] = post_tuple  
                if(global_pos_overlap is not None):
                    product_indices_np[k, global_pos_overlap] = self.overlap_volume_id
                k += 1
        
        product_indices_np = product_indices_np.astype(int)

        tuples = [tuple(row) for row in product_indices_np]
        self.index = pd.MultiIndex.from_tuples(tuples, names=global_index_names)

    def init_compute_data(self):
        self.pre_sites = np.atleast_1d(self.pre_features.pre_contact_sites.values).reshape(-1,1)
        self.post_sites = np.atleast_1d(self.post_features.post_contact_sites.values).reshape(1,-1)

        if("overlap_volume" in self.index.names):

            self.num_pre_sites = np.repeat(np.atleast_1d(self.pre_features.pre_contact_sites.values), self.post_features.shape[0])
            self.num_post_sites = np.tile(np.atleast_1d(self.post_features.post_contact_sites.values), self.pre_features.shape[0])

            self.M_synaptic_pairs = (self.pre_sites @ self.post_sites)
            self.M_null = self.M_synaptic_pairs / self.num_synapses

            self.M_null_flat = self.M_null.flatten()
            self.M_synaptic_pairs_flat = self.M_synaptic_pairs.flatten()
            
            self.index, self.M_null_flat, self.M_null, self.M_synaptic_pairs_flat, self.num_pre_sites, self.num_post_sites = correct_self_connections(self)
        else:
            pass
            """
            a = self.pre_features.pre_contact_sites.values
            a = a / a.sum()
            b = self.post_features.post_contact_sites.values
            b = b / b.sum()

            self.a = jnp.array(a)
            self.b = jnp.array(b)
            """

   
        
    def compute(self, model_df):
        pass
        """
        cost_matrix = model_df.loc[self.index, "cost"].values.reshape((self.a.size, self.b.size))           
        #print("C", cost_matrix, np.max(cost_matrix))
        cost_matrix_jnp = jnp.array(cost_matrix)
        
        geom = geometry.Geometry(cost_matrix_jnp)
        prob = linear_problem.LinearProblem(geom, self.a, self.b)
        solver = sinkhorn.Sinkhorn()
        output = solver(prob)
        matrix = output.matrix

        model_df.loc[self.index, "expected_synapse_count"] = matrix.flatten() * self.num_synapses
        """
        
    def compute_batch(self, model_dfs):
        pass
        """
        num_samples = len(model_dfs)

        cost_matrices = np.zeros((num_samples, self.a.size, self.b.size))
        for sample_idx in range(0, num_samples):
            cost_matrix = model_dfs[sample_idx].loc[self.index, "cost"].values.reshape((self.a.size, self.b.size))             
            cost_matrices[sample_idx] = cost_matrix
        
        C_batch = jnp.array(cost_matrices)
        
        solver = sinkhorn.Sinkhorn(max_iterations=500)
        
        def solve_ot(C):
            geom = geometry.Geometry(cost_matrix=C)
            problem = linear_problem.LinearProblem(geom, self.a, self.b)
            out = solver(problem)
            return out.matrix

        solve_ot_vmap = vmap(solve_ot)
        solve_ot_vmap_jit = jit(solve_ot_vmap)

        results = solve_ot_vmap_jit(C_batch)

        for sample_idx in range(0, num_samples):
            model_dfs[sample_idx].loc[self.index, "expected_synapse_count"] = results[sample_idx].flatten() * self.num_synapses
        """


class IndexData:
    def __init__(self):
        
        self.overlap_volumes = None
        self.global_index = None
        self.global_template_df = None

    def build_global_index(self, df_synapses_indexed):
        
        unioned_index_tuples = set()
        for overlap_volume in self.overlap_volumes:
            unioned_index_tuples = unioned_index_tuples.union(set(overlap_volume.index))
        
        global_index = pd.MultiIndex.from_tuples(list(unioned_index_tuples), names=df_synapses_indexed.index.names) 
        self.global_index = global_index.sort_values()

        mapped_pre_ids = np.unique(self.global_index.get_level_values("pre_id_mapped").values) 
        mapped_post_ids = np.unique(self.global_index.get_level_values("post_id_mapped").values)

        self.num_ids_mapped = np.max(np.union1d(mapped_pre_ids, mapped_post_ids)) + 1
        self.num_neuron_pairs = self.num_ids_mapped * (self.num_ids_mapped -1)
        self.neuron_indices = self.get_mapped_neuron_indices() # indices of the mapped neurons in the global index 

    def get_overlap_volume(self, overlap_volume_id):
        for overlap_volume in self.overlap_volumes:
            if(overlap_volume.overlap_volume_id == overlap_volume_id):
                return overlap_volume

        raise ValueError(overlap_volume_id)

    def get_model_df(self):
        model_df = pd.DataFrame(index=self.global_index, columns=[MODEL_REFERENCE, MODEL_CURRENT], dtype=float)         
        return model_df
    
    def get_summary_df(self, df_synapses_indexed):
        #summary_df = pd.DataFrame(index=self.global_index, columns=[EMPIRICAL, MODEL_NULL, MODEL_P, MODEL_C, MODEL_S_EXC, MODEL_S_INH,  MODEL_S, MODEL_PC, MODEL_PS, MODEL_CS, MODEL_PCS], dtype=float)
        summary_df = pd.DataFrame(index=self.global_index, columns=[EMPIRICAL, MODEL_NULL], dtype=float)
        summary_df.loc[df_synapses_indexed.index, EMPIRICAL] = df_synapses_indexed.synapse_count.values
        summary_df.loc[summary_df[EMPIRICAL].isna(), EMPIRICAL] = float(0)
        return summary_df
    
    def get_mapped_neuron_indices(self, additional_groupby_fields = []):
        groupby_fields = ["pre_id_mapped", "post_id_mapped"] 
        groupby_fields.extend(additional_groupby_fields)
        indices = self.get_model_df().groupby(groupby_fields).indices
        return {k: v for k, v in indices.items() if k[0] != -1 and k[1] != -1}


def compile_index_data(eval_folder, df_synapses_indexed):
    index_file = eval_folder / 'index_data.pkl'
    if index_file.exists():
        with open(index_file, 'rb') as file:
            index_data = pickle.load(file)
        return index_data
    else: 
        overlap_volumes = []
        for overlap_volume_id, synapses_overlap_volume in df_synapses_indexed.groupby("overlap_volume"):
            overlap_volumes.append(OverlapVolume(overlap_volume_id, 
                                                get_local_pre_features(synapses_overlap_volume), 
                                                get_local_post_features(synapses_overlap_volume),
                                                df_synapses_indexed.index.names))
        index_data = IndexData()
        index_data.overlap_volumes = overlap_volumes
        index_data.build_global_index(df_synapses_indexed)

        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)

        return index_data