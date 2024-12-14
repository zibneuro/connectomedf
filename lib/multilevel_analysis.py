from .constants import MODEL_NULL, EMPIRICAL
from tqdm.notebook import tqdm

class MultilevelAnalysis:
    def __init__(self, index_data, df_synapses_indexed, statistics, num_realizations=100):
        self.index_data = index_data
        self.df_summary = index_data.get_summary_df(df_synapses_indexed)
        self.stats = statistics
        self.num_realizations = num_realizations
        
    def run_null_and_empirical(self):

        for overlap_volume in self.index_data.overlap_volumes:
            self.df_summary.loc[overlap_volume.index, MODEL_NULL] = overlap_volume.M_null_flat
            self.df_summary.loc[overlap_volume.index, "num_synaptic_pairs"] = overlap_volume.M_synaptic_pairs_flat.astype(int)
            self.df_summary.loc[overlap_volume.index, "num_pre_sites"] = overlap_volume.num_pre_sites.astype(int)
            self.df_summary.loc[overlap_volume.index, "num_post_sites"] = overlap_volume.num_post_sites.astype(int)

        self.df_summary.loc[:, f"{MODEL_NULL}_group_index"] = self.df_summary.index.get_level_values("overlap_volume").values
        self.df_summary.loc[:, f"{MODEL_NULL}_preference"] = 1

        self.stats.compute(self.df_summary, EMPIRICAL, EMPIRICAL)
        self.stats.compute(self.df_summary, MODEL_NULL, MODEL_NULL, realize=True, num_realizations=self.num_realizations)

    def run_model(self, 
            reference_model_descriptor,
            model,
            model_descriptor):
        
        assert isinstance(reference_model_descriptor, str)
        assert isinstance(model_descriptor, str)
                   
        group_index_column_reference = f"{reference_model_descriptor}_group_index"
        if(group_index_column_reference not in self.df_summary.index.names):
            self.df_summary.set_index(group_index_column_reference, append=True, inplace=True)

        values_model, specificity_model, group_indices_model = model.compute(self.df_summary, reference_model_descriptor, group_index_column_reference)

        self.df_summary.reset_index(group_index_column_reference, inplace=True)

        self.df_summary.loc[:, model_descriptor] = values_model
        self.df_summary.loc[:, f"{model_descriptor}_preference"] = specificity_model
        self.df_summary.loc[:, f"{model_descriptor}_group_index"] = group_indices_model

        self.stats.compute(self.df_summary, model_descriptor, model_descriptor, realize=True, num_realizations=self.num_realizations)

    def run_model_with_parameters(self,
                                reference_model_descriptor,
                                model,
                                model_descriptor,
                                parameter_values):
        
        self.stats.delete_by_model(model_descriptor)

        group_index_column_reference = f"{reference_model_descriptor}_group_index"

        for param_idx in tqdm(range(parameter_values.shape[0])):
            params = parameter_values[param_idx]

            if(group_index_column_reference not in self.df_summary.index.names):
                self.df_summary.set_index(group_index_column_reference, append=True, inplace=True)

            values_model, specificity_model, group_indices_model = model.compute(self.df_summary, reference_model_descriptor, params, group_index_column_reference)

            self.df_summary.reset_index(group_index_column_reference, inplace=True)

            # record model prediction
            self.df_summary.loc[:, model_descriptor] = values_model
            self.df_summary.loc[:, f"{model_descriptor}_preference"] = specificity_model
            self.df_summary.loc[:, f"{model_descriptor}_group_index"] = group_indices_model

            self.stats.compute(self.df_summary, model_descriptor, model_descriptor, realize=True, parameter_index=param_idx)