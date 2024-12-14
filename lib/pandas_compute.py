import numpy as np

from .constants import *

def compute_loss(df, probability_column):
    loss = -np.log(df[probability_column]).values
    return loss

def get_delta_loss_column(model_reference, model_target):
    return f"delta_loss_{model_reference}_{model_target}"

def get_loss_column(model):
    return f"loss_{model}"

def compute_observation_probability(empirical, model_values):
    mask_connected = empirical != 0
    p_unconnected = np.exp(-model_values)
    p_connected = 1 - p_unconnected

    observation_probability = np.zeros(len(empirical))
    observation_probability[mask_connected] = p_connected[mask_connected]
    observation_probability[~mask_connected] = p_unconnected[~mask_connected]
    return observation_probability


def compute_delta_loss(df, model_reference, model_target):

    reference_probability_column = model_reference + "_observation_probability"
    if(reference_probability_column not in df.columns):
        df.loc[:, reference_probability_column] = compute_observation_probability(df[EMPIRICAL].values, df[model_reference].values)

    target_probability_column = model_target + "_observation_probability"
    if(target_probability_column not in df.columns):
        df.loc[:, target_probability_column] = compute_observation_probability(df[EMPIRICAL].values, df[model_target].values)

    loss_reference = compute_loss(df, reference_probability_column)
    loss_target = compute_loss(df, target_probability_column)
    
    reference_loss_column = get_loss_column(model_reference)
    if(reference_loss_column not in df.columns):
        df.loc[:, reference_loss_column] = loss_reference

    target_loss_column = get_loss_column(model_target)
    if(target_loss_column not in df.columns):
        df.loc[:, target_loss_column] = loss_target

    df.loc[:, get_delta_loss_column(model_reference, model_target)] = loss_reference - loss_target


def get_df_cellular(df, selected_models, separate_compartment=False, excluded_neuron_ids = [],
        pre_celltype_column="pre_celltype", post_celltype_column="post_celltype"):
    
    aggregation_dict = {
        EMPIRICAL : "sum"
    }
    for model in selected_models:
        aggregation_dict[model] = "sum"
    if(separate_compartment):
        aggregation_columns = ["pre_id_mapped", pre_celltype_column, "post_id_mapped", post_celltype_column, "post_compartment"]
    else:
        aggregation_columns = ["pre_id_mapped", pre_celltype_column, "post_id_mapped", post_celltype_column]
    
    df_per_pair = df.groupby(aggregation_columns).agg(aggregation_dict)

    for model in selected_models:
        df_per_pair.loc[:, model + "_observation_probability"] = compute_observation_probability(df_per_pair[EMPIRICAL], df_per_pair[model])
    
    if(len(excluded_neuron_ids)):
        mask_excluded = df_per_pair.index.get_level_values("pre_id_mapped").isin(excluded_neuron_ids) | df_per_pair.index.get_level_values("post_id_mapped").isin(excluded_neuron_ids)
        df_per_pair = df_per_pair[~mask_excluded]
    return df_per_pair

def get_delta_syncount_column(model_reference, model_target):
    return f"delta_syncount_{model_reference}_{model_target}"

def compute_delta_syncount(df, reference_model, target_model):
    assert reference_model in df.columns, f"{reference_model} not in columns"
    assert target_model in df.columns, f"{target_model} not in columns"

    df.loc[:, get_delta_syncount_column(reference_model, target_model)] = df[target_model] - df[reference_model]