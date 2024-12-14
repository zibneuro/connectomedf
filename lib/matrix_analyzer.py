import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib as mpl
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .util_plot import resize_image, figsize_mm_to_inch, NormalizePreferenceValue, NormalizeDeviationValue
from .constants import EMPIRICAL

def get_neuron_to_neuron_domain(df_summary, 
                                pre_celltype_column = "pre_celltype_merged",
                                post_celltype_column = "post_celltype_merged",
                                return_neuron_ids_only = False,
                                celltype_order = None,
                                ignored_neuron_ids = [-1]): 
    celltype_dictionary = {}

    def register_celltypes(indices):
        for entry in indices:
            ct, neuron_id = entry
            if(neuron_id in ignored_neuron_ids):
                continue
            if(neuron_id in celltype_dictionary):
                try:
                    assert celltype_dictionary[neuron_id] == ct
                except:
                    raise ValueError(f"Neuron id {neuron_id} has multiple celltypes: {celltype_dictionary[neuron_id]} and {ct}")
            else:
                celltype_dictionary[neuron_id] = ct

    row_indices = df_summary.groupby([pre_celltype_column, "pre_id_mapped"]).indices.keys()
    register_celltypes(row_indices)
    col_indices = df_summary.groupby([post_celltype_column, "post_id_mapped"]).indices.keys() 
    register_celltypes(col_indices)

    celltype_dictionary
    domain = [(item[1], item[0]) for item in celltype_dictionary.items() if item[0] not in ignored_neuron_ids]
    
    if(celltype_order is None):
        domain_sorted = sorted(domain, key=lambda x: x[0])
    else:
        for ct in celltype_dictionary.values():
            if ct not in celltype_order:
                raise ValueError(f"Celltype {ct} not in celltype_order")
        domain_sorted = sorted(domain, key=lambda x: celltype_order.index(x[0]))
    
    if(return_neuron_ids_only):
        return [item[1] for item in domain_sorted]
    else:
        return domain_sorted  


class ConnectomeMatrixAnalyzer:
    
    def __init__(self, 
                 df_summary : pd.DataFrame,
                 working_folder : Path):
        
        self.df_summary = df_summary

        self.working_folder = working_folder
        self.working_folder.mkdir(parents=True, exist_ok=True)        
        
        # assert EMPIRICAL in df_summary.columns
        self.data_column_A = EMPIRICAL
        self.data_column_B = None

        self.init_colormaps()
        
    def init_colormaps(self):
        self.colormaps = {}
        
        self.colormaps["viridis"] = plt.colormaps['viridis'].copy()
        self.colormaps["viridis_reversed"] = plt.colormaps['viridis'].copy().reversed()
        self.colormaps["blues"] = plt.colormaps['Blues'].copy()
        self.colormaps["blues"].set_bad(color='white')
        self.colormaps["reds"] = plt.colormaps['Reds'].copy()
        self.colormaps["reds"].set_bad(color='white')
        self.colormaps["binary"] = plt.colormaps['binary'].copy()
        self.colormaps["coolwarm"] = plt.colormaps['coolwarm'].copy()
        self.colormaps["BrBG"] = plt.colormaps["BrBG"].copy()
        self.colormaps["yellowbrown"] = plt.colormaps["YlOrBr"].copy()
        self.colormaps["gist_heat"] = plt.colormaps["gist_heat"].copy()
        self.colormaps["gist_heat_reversed"] = plt.colormaps["gist_heat"].copy().reversed()
    
    def set_data_columns(self, data_column_A):
        assert data_column_A in self.df_summary.columns
        self.data_column_A = data_column_A

    def set_selection(self, 
                      include_filter_or = {},
                      include_filter_and = {}, 
                      exclude_filter = {}):
        
        def get_mask(filter_index_name, filter_values):
            assert filter_index_name in self.df_summary.index.names
            return self.df_summary.index.get_level_values(filter_index_name).isin(filter_values)

        def get_include_mask(filter_spec, mode="or"):
            if(not len(filter_spec)):
                return np.ones(len(self.df_summary), dtype=bool)
            else:
                if(mode == "or"):   
                    mask_include = np.zeros(len(self.df_summary), dtype=bool)
                    for filter_index_name, filter_values in include_filter_or.items():
                        mask_include = mask_include | get_mask(filter_index_name, filter_values)
                    return mask_include
                elif(mode == "and"):
                    mask_include = np.ones(len(self.df_summary), dtype=bool)
                    for filter_index_name, filter_values in include_filter_and.items():
                        mask_include = mask_include & get_mask(filter_index_name, filter_values)
                    return mask_include
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                
        def get_exclude_mask(filter_spec):
            mask_exclude = np.zeros(len(self.df_summary), dtype=bool)
            for filter_index_name, filter_values in filter_spec.items():
                mask_exclude = mask_exclude | get_mask(filter_index_name, filter_values)
            return mask_exclude
                
        mask_include_or = get_include_mask(include_filter_or, mode="or")
        mask_include_and = get_include_mask(include_filter_and, mode="and")
        mask_exclude = get_exclude_mask(exclude_filter)
        filter_mask = mask_include_or & mask_include_and & ~mask_exclude

        df_filtered = self.df_summary.loc[filter_mask].copy()
        self.df_filtered = df_filtered

    def rebuild_matrix(self):
        assert self.last_parameters is not None
        self.build_matrix(**self.last_parameters)    

    def build_matrix(self,
                    row_attributes,
                    col_attributes,
                    row_domains = None,
                    col_domains = None,
                    value_label_map = {}, # nested dict: attribute_name -> {attribute_value : label}
                    aggregation_fn = "sum",
                    default_value = 0):
        
        self.last_parameters = {
            "row_attributes" : row_attributes,
            "col_attributes" : col_attributes,
            "row_domains" : row_domains,
            "col_domains" : col_domains,
            "value_label_map" : value_label_map,
            "aggregation_fn" : aggregation_fn,
            "default_value" : default_value
        }
        
        def assert_format(attributes, domains):
            # check / convert attributes format
            if(not isinstance(attributes, list)):
                attributes_list = [attributes]
            else:
                attributes_list = attributes
            for att in attributes_list:
                assert isinstance(att, str)
                try:
                    assert att in self.df_filtered.index.names
                except:
                    raise ValueError(f"Attribute {att} not in index names: {self.df_filtered.index.names}")

            # check / convert domains format
            if(domains is None):
                return attributes_list, None
            
            if(isinstance(domains, np.ndarray)):
                assert domains.ndim == 1
                domains = domains.tolist()

            assert isinstance(domains, list)
            
            if(len(attributes_list) > 1):
                for item in domains:
                    assert isinstance(item, tuple)
                    assert len(item) == len(attributes_list)
                domains_list = domains
            else:
                domains_list = []
                for item in domains:
                    assert isinstance(item, int) or isinstance(item, str) or isinstance(item, np.integer)
                    domains_list.append((item,)) 

            return attributes_list, domains_list
        
        row_attributes, row_domains = assert_format(row_attributes, row_domains)
        col_attributes, col_domains = assert_format(col_attributes, col_domains)
 
        self.df_aggregated = self.df_filtered.groupby(row_attributes + col_attributes).aggregate({self.data_column_A : aggregation_fn})
        row_indices = self.df_filtered.groupby(row_attributes).indices
        col_indices = self.df_filtered.groupby(col_attributes).indices

        assert len(row_indices)
        assert len(col_indices)

        def force_key_to_tuple(key):
            if(isinstance(key, tuple)):
                return key
            elif(isinstance(key, int) or isinstance(key, np.integer)):
                return (key,)
            elif(isinstance(key, list)):
                return tuple(key)
            else:
                raise TypeError(key)

        if(row_domains is None):
            row_domains = [force_key_to_tuple(key) for key in sorted(row_indices.keys())]
        if(col_domains is None):
            col_domains = [force_key_to_tuple(key) for key in sorted(col_indices.keys())]

        row_meta = {
            "attributes" : row_attributes,
            "domains" : row_domains,
            "annotations" : [] # [(level-1-label, level-2-label, ...), ...]
        } 
        col_meta = {
            "attributes" : col_attributes,
            "domains" : col_domains,
            "annotations" : [] # [(level-1-label, level-2-label, ...), ...]
        }

        def get_annotations(current_key, attribute_names):
            assert isinstance(current_key, tuple)
            assert len(current_key) == len(attribute_names)
            annotations = []
            for nesting_level in range(len(attribute_names)):
                attribute_name = attribute_names[nesting_level]
                current_value = current_key[nesting_level]
                if(attribute_name in value_label_map):
                    label_map = value_label_map[attribute_name]
                    if(current_value in label_map):
                        annotations.append(str(label_map[current_value]))
                    else:
                        annotations.append(str(current_value))        
                else:
                    annotations.append(str(current_value))
            return tuple(annotations)

        M = default_value * np.ones((len(row_domains), len(col_domains)))
   
        for i in range(len(row_domains)):
            row_key = row_domains[i]
            for j in range(len(col_domains)):   
                col_key = col_domains[j]    
                index_key = row_key + col_key                    
                if(index_key in self.df_aggregated.index):
                    M[i,j] = self.df_aggregated.loc[index_key].values[0]
                
                if(i == 0):
                    col_meta["annotations"].append(get_annotations(col_key, col_attributes))
            row_meta["annotations"].append(get_annotations(row_key, row_attributes)) 
            
        self.matrix = M
        self.row_meta = row_meta
        self.col_meta = col_meta


    def render_matrix(self, 
                      plot_name : str, 
                      colormap_name = "binary", vmin=-1, vmax=1,
                      normalization_function = None,
                      row_markers = {}, # attribute_name -> [attribute_value, ...]
                      col_markers = {}, # attribute_name -> [attribute_value, ...]  
                      high_res = False, 
                      row_separator_lines = True,
                      col_separator_lines = True):
        
        assert self.matrix is not None
        assert self.row_meta is not None
        assert self.col_meta is not None

        M = self.matrix.copy()
        row_meta = self.row_meta
        col_meta = self.col_meta

        use_mpl_norm = (normalization_function is not None) and isinstance(normalization_function, Normalize)
        if((normalization_function is not None) and not use_mpl_norm): 
            M = normalization_function(M)

        assert "png" not in plot_name
        assert colormap_name in self.colormaps
        colormap = self.colormaps[colormap_name]    
        
        outfolder = self.working_folder / plot_name 
        outfolder.mkdir(parents=True, exist_ok=True)

        # configure matplotlib settings
        mpl.rcParams["axes.spines.top"] = True
        mpl.rcParams["axes.spines.right"] = True
        mpl.rcParams['font.size'] = 5   
        mpl.rcParams["legend.frameon"] = False
        mpl.rcParams["legend.fontsize"] = 7
        mpl.rcParams["font.family"] = "Arial"    
        mpl.rcParams["axes.labelsize"] = 7
        mpl.rcParams["xtick.labelsize"] = 7
        mpl.rcParams["ytick.labelsize"] = 6
        mpl.rcParams["mathtext.default"] = "regular"
        mpl.rcParams["figure.max_open_warning"] = 0
        
        mpl.use("cairo")
        
        def get_minortick_locations(meta, markers):
            locations = []
            labels_int = []
            labels_str = []
            
            for attribute_name, attribute_values in markers.items():
                if(attribute_name in meta["attributes"]):
                    attribute_index = meta["attributes"].index(attribute_name)
                    for k in range(len(meta["domains"])):
                        current_key = meta["domains"][k]
                        attribute_value = current_key[attribute_index]
                        if(attribute_value in attribute_values):
                            locations.append(k+0.5)
                            labels_int.append(len(labels_int)+1)
                            labels_str.append(str(attribute_value))
            
            return locations, labels_int, labels_str 


        n = M.shape[0]
        m = M.shape[1]
        extent = (0, m, n, 0)

        plt.clf()
        fig = plt.figure(figsize=figsize_mm_to_inch(120,60))
        fig.subplots_adjust(left=0.05, right=0.5, bottom=0.1, top=0.9)
        ax = fig.add_subplot()

        if(use_mpl_norm):   
            pos = ax.imshow(M, extent=extent, cmap=colormap, norm=normalization_function, interpolation='none')
        else:
            pos = ax.imshow(M, extent=extent, cmap=colormap, vmin=vmin, vmax=vmax, interpolation='none')
        cbar = fig.colorbar(pos, ax=ax, fraction=0.05, pad=0.05)

        if(isinstance(normalization_function, NormalizePreferenceValue)):
            assert vmin == -1
            assert vmax == 1
            cbar.set_ticks([-1, 0, 1])  
            min_value = normalization_function.min_value - 1
            mid_value = normalization_function.mid_value - 1
            max_value = normalization_function.max_value - 1
            cbar.set_ticklabels([f'{min_value:.1f}', f'{mid_value:.1f}', f'{max_value:.1f}'])
        elif(isinstance(normalization_function, NormalizeDeviationValue)):
            assert vmin == 0
            assert vmax == 1
            cbar.set_ticks([0, 0.5, 1])  
            min_value = normalization_function.min_value
            mid_value = normalization_function.mid_value
            max_value = normalization_function.max_value
            cbar.set_ticklabels([f'{min_value:.1f}', f'{mid_value:.1f}', f'{max_value:.1f}'])

        if(high_res):
            HIGH_RES_LINEWIDTH = 0.1
            ax.grid(color='black', linewidth=HIGH_RES_LINEWIDTH)
            ax.set_frame_on(True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(HIGH_RES_LINEWIDTH)
            ax.tick_params(axis='both', which='both', width=HIGH_RES_LINEWIDTH)
        else:
            ax.grid(color='black', linewidth=0.5)
            ax.set_frame_on(True)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(1)

        def get_separator_line_positions(meta):
            annotations = meta["annotations"]
            nesting_level = 0
            last_label = annotations[0][nesting_level]
            positions = []
            for k in range(1, len(annotations)):
                current_label = annotations[k][nesting_level]
                if(current_label != last_label):
                    positions.append(k)
                    last_label = current_label
            return positions

        # write meta information
        if(row_separator_lines):
            pos_rows = get_separator_line_positions(row_meta)
            ax.set_yticks(pos_rows)
        else:
            ax.set_yticks([])
        ax.set_yticklabels([])

        if(col_separator_lines):
            pos_cols = get_separator_line_positions(col_meta)
            ax.set_xticks(pos_cols)
        else:
            ax.set_xticks([]) 
        ax.set_xticklabels([])
        
        row_locations, row_labels, row_labels_str = get_minortick_locations(row_meta, row_markers)
        col_locations, col_labels, col_labels_str = get_minortick_locations(col_meta, col_markers)        
        ax.set_xticks(col_locations, minor=True)
        ax.set_xticklabels(col_labels, minor=True)
        ax.set_yticks(row_locations, minor=True)
        ax.set_yticklabels(row_labels, minor=True)
        
        ax.tick_params(axis='both', which='minor', color="red")
        
        def write_lines(lines):
            x = 0.6
            y = 0.9
            dy = -0.04
            for line in lines:
                plt.text(x, y, line, transform=fig.transFigure, va='center')
                y += dy

        def write_title(title):
            x = 0.06
            y = 0.93
            plt.text(x, y, title, transform=fig.transFigure, va='center')
        
        def write_selections_info(meta, max = 8):
            
            def get_annotations(nesting_level):
                if(nesting_level > len(meta["attributes"])):
                    return get_annotations(len(meta["attributes"]))
      
                annotations = []
                for i in range(len(meta["annotations"])):
                    current = " | ".join(list(meta["annotations"][i])[0:nesting_level])
                    if(not len(annotations)):
                        annotations.append(current)
                    elif(annotations[-1] != current):
                        annotations.append(current)

                return annotations

            level1_annotations = get_annotations(1)
            level2_annotations = get_annotations(2)

            if(len(level2_annotations) <= max):
                lines = level2_annotations
            else:
                lines = level1_annotations

            if(len(lines) > max):
                lines = lines[0:max] + ["..."]
            return lines

        write_title(f"{plot_name} | {self.data_column_A}")
        
        def write_marker_info(labels, labels_str):
            lines = []
            lines.append("; ".join([f"{labels[k]}: {labels_str[k]}" for k in range(len(labels))]))
            return lines

        lines = []
        row_atts = " | ".join(row_meta["attributes"])
        lines += [f"rows ({M.shape[0]}): {row_atts}"]
        lines += write_selections_info(row_meta)
        col_atts = " | ".join(col_meta["attributes"])
        lines += [f"cols ({M.shape[1]}): {col_atts}"]
        lines += write_selections_info(col_meta)
    
        lines += ["row markers:"]
        lines+= write_marker_info(row_labels, row_labels_str)
        lines += ["col markers:"]
        lines+= write_marker_info(col_labels, col_labels_str)
        
        write_lines(lines)
        
        fig.savefig(outfolder/f"{plot_name}.png", dpi=500)    
        image = Image.open(outfolder/f"{plot_name}.png")
        fig.savefig(outfolder/f"{plot_name}.pdf", transparent=True)
        if(high_res):
            fig.savefig(outfolder/f"{plot_name}_high_res.png", dpi=2000)

        return resize_image(image)