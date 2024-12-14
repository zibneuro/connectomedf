
from lib.constants import *

import numpy as np

class Model:
    def __init__(self, index_data, groupby_fields = [], sequential=True, norm_specificity = False):
        self.index_data = index_data
        self.groupby_fields = groupby_fields
        self.sequential = sequential
        self.norm_specificity = norm_specificity

    def compute(self, df_summary, reference_model_descriptor, group_index_column_reference):
        assert len(self.groupby_fields)

        values_empirical = df_summary[EMPIRICAL].values
        values_reference_model = df_summary[reference_model_descriptor].values

        if(self.sequential):
            groupby_fields = self.groupby_fields + [group_index_column_reference]
        else:
            groupby_fields = self.groupby_fields
        indices = df_summary.groupby(groupby_fields).indices
         
        specificity_values = np.ones(len(df_summary))
        values_model = values_reference_model.copy()
        group_indices_model = np.ones(len(df_summary)).astype(int)

        def norm_specificity_value(specificity_value, sum_model, sum_total):
            sum_new = specificity_value * sum_model
            delta = sum_new - sum_model
            return delta / sum_total

        def get_overlap_volume_indices(df_summary, global_indices):
            overlap_volumes = df_summary.index.get_level_values('overlap_volume')
            unique_volumes = set(overlap_volumes[global_indices])
            if len(unique_volumes) > 1:
                raise ValueError("Multiple unique overlap volumes found.")
            current_overlap_volume = unique_volumes.pop()
            return current_overlap_volume, np.where(overlap_volumes == current_overlap_volume)[0]

        # iterate over groupings
        group_index = 0
        for group_key, global_indices in indices.items():
            group_indices_model[global_indices] = group_index
            
            # empirically observed synapse count for this group
            sum_empirical = values_empirical[global_indices].sum()
            # synapse count predicted for this group in reference model
            sum_model = values_reference_model[global_indices].sum()

            specificity_value = 1
            if(sum_model > 0):
                specificity_value = sum_empirical / sum_model
            elif(sum_empirical > 0):
                raise RuntimeError(group_key)    
            
            expected_syncounts = specificity_value * values_reference_model[global_indices]

            values_model[global_indices] = expected_syncounts

            if(self.norm_specificity):
                overlap_id, overlap_volume_indices = get_overlap_volume_indices(df_summary, global_indices)
                sum_total = values_empirical[overlap_volume_indices].sum()
                normed_spec_value = norm_specificity_value(specificity_value, sum_model, sum_total)
                specificity_values[global_indices] = normed_spec_value
            else:
                specificity_values[global_indices] = specificity_value

            group_index += 1
                
        return values_model, specificity_values, group_indices_model

    def get_prior(self):
        raise NotImplementedError
    
    def get_synapse_vector_mapped_neurons(self, synapse_values):
        raise NotImplementedError
    

class NullModel(Model):
    def __init__(self, index_data):
        super().__init__(index_data)
    
    def compute(self, model_df):        
        model_df[MODEL_CURRENT] = float(0)
        for overlap_volume in self.index_data.overlap_volumes:
            model_df.loc[overlap_volume.index, MODEL_CURRENT] += overlap_volume.M_null_flat