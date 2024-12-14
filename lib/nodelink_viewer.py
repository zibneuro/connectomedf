from .util_plot import rgb_to_js_color
from pyvis.network import Network
from pyvis.options import Layout


class DefaultNodeStyler():
    def __init__(self, excitatory_type_ids, inhibitory_type_ids, node_color = "dimgrey"):
        self.excitatory_type_ids = excitatory_type_ids
        self.inhibitory_type_ids = inhibitory_type_ids
        self.default_color = node_color

    def get_shape(self, celltype_id):
            if(celltype_id in self.excitatory_type_ids):
                return "triangle"
            elif(celltype_id in self.inhibitory_type_ids):
                return "circle"
            else:
                raise ValueError(celltype_id)
    
    def get_style(self, is_connected, neuron_id, celltype_id):
        return self.get_shape(celltype_id), self.default_color


class PotentialConnectionsNodeStyler(DefaultNodeStyler):
    def __init__(self, excitatory_type_ids, inhibitory_type_ids,
                  connected_color = "black", unconnected_color = "lightgrey", highlighted_colors = {}):      
        super().__init__(excitatory_type_ids, inhibitory_type_ids)

        self.connected_color = connected_color
        self.unconnected_color = unconnected_color
        self.highlighted_colors = highlighted_colors

    def get_style(self, is_connected, neuron_id, celltype_id):        
        shape = self.get_shape(celltype_id)

        if(neuron_id in self.highlighted_colors):
            color = self.highlighted_colors[neuron_id]
        elif(is_connected):
            color = self.connected_color      
        else:      
            color = self.unconnected_color

        return shape, color        


class DefaultEdgeStyler():
    def __init__(self, dendrite_id, soma_id, ais_id):
        self.dendrite_id = dendrite_id
        self.soma_id = soma_id
        self.ais_id = ais_id

    def reset(self):
        pass

    def get_color(self, compartment_id):
        if(compartment_id == self.dendrite_id):
            return "blue"
        elif(compartment_id == self.soma_id):
            return "red"
        elif(compartment_id == self.ais_id):
            return "orange"
        else:
            raise ValueError(compartment_id)

    def get_style(self, pre_id, post_id, synapse_count, value, post_compartment, neurons_connected):
        if(pre_id == post_id):
            return None, None

        return self.get_color(post_compartment), str(synapse_count)


class PotentialConnectionsEdgeStyler(DefaultEdgeStyler):
    def __init__(self, dendrite_id, soma_id, ais_id, connected_color = "black", unconnected_color = "lightgrey", 
            only_highlighted_multiedge = False, only_connected_multiedge=False, no_multiedge=False, highlighted_colors = {},
            syncount_labels=False, compartment_labels=True):
        super().__init__(dendrite_id, soma_id, ais_id)

        self.connected_color = connected_color
        self.unconnected_color = unconnected_color
        self.syncount_labels = syncount_labels 
        self.compartment_labels = compartment_labels        
        self.highlighted_colors = highlighted_colors
        self.only_highlighted_multiedge = only_highlighted_multiedge
        self.only_connected_multiedge = only_connected_multiedge
        self.no_multiedge = no_multiedge
        if(self.only_connected_multiedge):
            assert not only_highlighted_multiedge    
        self.walked_pairs = set()

    def reset(self):
        self.walked_pairs = set()

    def get_style(self, pre_id, post_id, synapse_count, value, post_compartment, neurons_connected):
        if(pre_id == post_id):
            return None, None

        is_connected = synapse_count > 0

        if(neurons_connected and not is_connected):
            return None, None

        if(is_connected):
            if(self.no_multiedge):
                color = self.connected_color
            elif (post_id not in self.highlighted_colors and self.only_highlighted_multiedge):            
                color = self.connected_color            
            elif(post_compartment == self.dendrite_id):
                color = "blue"
            elif(post_compartment == self.soma_id):
                color = "red"
            elif(post_compartment == self.ais_id):
                color = "orange"
            else:
                raise ValueError(post_compartment)
        else:
            color = self.unconnected_color

        label = ""
        if(post_compartment == -1):
            if(self.syncount_labels and synapse_count >= 1):
                label = str(int(synapse_count))            
        elif(self.compartment_labels):
            if(post_compartment == self.dendrite_id):
                label = "D"
            elif(post_compartment == self.soma_id):
                label = "S"
            elif(post_compartment == self.ais_id):
                label = "A"
            else:
                raise ValueError()

        if(self.syncount_labels and is_connected):
            syncount_str = str(int(synapse_count))
            if(self.compartment_labels):
                label += f"({syncount_str})"
            else:
                label = syncount_str         

        if(label == ""):
            label = None

        if(self.no_multiedge):
            if((pre_id, post_id) in self.walked_pairs):            
                return None, None
        elif(self.only_highlighted_multiedge and post_id not in self.highlighted_colors and not is_connected):
            if((pre_id, post_id) in self.walked_pairs):            
                return None, None
            else:
                label = ""            
        elif(self.only_connected_multiedge and not is_connected):
            if((pre_id, post_id) in self.walked_pairs):            
                return None, None   

        self.walked_pairs.add((pre_id, post_id))
        return color, label


class SpecificityEdgeStyler(PotentialConnectionsEdgeStyler):
    def __init__ (self, dendrite_id, soma_id, ais_id, color_interpolator,
            syncount_labels=False, compartment_labels=True, only_highlighted_multiedge = False, highlighted_colors = {}):

        super().__init__(dendrite_id, soma_id, ais_id, syncount_labels=syncount_labels, compartment_labels=compartment_labels,
            only_highlighted_multiedge = only_highlighted_multiedge, highlighted_colors = highlighted_colors)
        
        self.color_interpolator = color_interpolator


    def get_style(self, pre_id, post_id, synapse_count, value, post_compartment, neurons_connected):
        if(pre_id == post_id):
            return None, None
        
        if(self.only_highlighted_multiedge and post_id not in self.highlighted_colors):
            if((pre_id, post_id) in self.walked_pairs):            
                return None, None
        
        label = ""
        if(self.compartment_labels):
            if(post_compartment == self.dendrite_id):
                label = "D"
            elif(post_compartment == self.soma_id):
                label = "S"
            elif(post_compartment == self.ais_id):
                label = "A"
            else:
                raise ValueError(post_compartment)
        
        if(self.syncount_labels and synapse_count >= 1):
            syncount_str = str(int(synapse_count))
            label += f"({syncount_str})"

        rgba = self.color_interpolator(value)
        color  = rgb_to_js_color(rgba)

        self.walked_pairs.add((pre_id, post_id))
        return color, label




class SubnetworkVisualization():
    def __init__(self, plot_folder, node_styler, edge_styler, 
                 pre_id_column = "pre_id_mapped", post_id_column = "post_id_mapped", post_compartment_column = "post_compartment",
                 pre_celltype_column = "pre_celltype", post_celltype_column = "post_celltype"):
        
        self.plot_folder = plot_folder
        self.node_styler = node_styler
        self.edge_styler = edge_styler
        self.pre_id_column = pre_id_column
        self.post_id_column = post_id_column
        self.post_compartment_column = post_compartment_column
        self.pre_celltype_column = pre_celltype_column
        self.post_celltype_column = post_celltype_column
        self.single_presynaptic_id = None
        self.synapses_values = None


    def get_network_representation(self, df, synapse_column, value_column):

        pre_ids = df[self.pre_id_column].unique().tolist()
        if(len(pre_ids) == 1):
            single_presynaptic_id = pre_ids[0]
        else:
            single_presynaptic_id = None

        nodes = {}  # neuron_id -> (mapped_id, celltype)
        nodes_rev = {} # mapped_id -> neuron_id
        edges = {} # (mapped_id_pre, mapped_id_post) -> [(synapses, value, compartment)]

        connected_neurons = set()
        
        for _, row in df.iterrows():
            pre_id = row[self.pre_id_column]
            post_id = row[self.post_id_column]
            post_compartment = row[self.post_compartment_column]
            synapses = row[synapse_column]
            if(value_column is not None):
                value = row[value_column]
            else:
                value = 0

            if(synapses > 0):
                connected_neurons.add((pre_id, post_id))

            if(pre_id not in nodes):
                nodes[pre_id] = (len(nodes), row[self.pre_celltype_column])
                nodes_rev[len(nodes)-1] = pre_id

            if(post_id not in nodes):
                nodes[post_id] = (len(nodes), row[self.post_celltype_column])
                nodes_rev[len(nodes)-1] = post_id

            pair_key = (nodes[pre_id][0], nodes[post_id][0])
            if(pair_key not in edges):
                edges[pair_key] = []
            edges[pair_key].append((synapses, value, post_compartment)) 


        layout = Layout(randomSeed = 1000)
        layout.hierarchical = Layout.Hierarchical(enabled=False)
        nt = Network(directed=True, layout = layout)
        nt.show_buttons(filter_=["physics", "layout"])
        # all visjs options: https://github.com/visjs/vis-network/blob/master/lib/network/options.ts

        # add neurons as nodes
        for neuron_id, node in nodes.items():
            node_id = node[0]
            celltype_id = node[1]

            if(single_presynaptic_id is not None):
                is_connected = (single_presynaptic_id == neuron_id) or \
                    ((single_presynaptic_id, neuron_id) in connected_neurons)
            else:
                is_connected = False
            
            shape, color = self.node_styler.get_style(is_connected, neuron_id, celltype_id)
            nt.add_node(node_id, size=12, color=color, shape=shape, label=" ", title=str(neuron_id), physics=True)

        # add edges
        for edge_key, multiedge_parameters in edges.items():
            node_id_pre, node_id_post = edge_key 
            pre_id = nodes_rev[node_id_pre]
            post_id = nodes_rev[node_id_post]   
            neurons_connected = (pre_id, post_id) in connected_neurons
            for edge in multiedge_parameters: 
                synapse_count, value, post_compartment = edge
                edge_color, edge_label = self.edge_styler.get_style(pre_id, post_id, int(synapse_count), value, post_compartment, neurons_connected)   
                if(edge_color is not None):
                    nt.add_edge(node_id_pre, node_id_post, color=edge_color, label=edge_label)           
                
        nt.set_edge_smooth('dynamic')
        nt.options.layout.hierarchical.enabled = False
        #nt.barnes_hut()
        return nt


    def create(self, descriptor, df, synapse_column, value_column = None):
        self.edge_styler.reset()

        network = self.get_network_representation(df, synapse_column, value_column)
        
        html_string = network.generate_html()
        html_file = self.plot_folder/"{}.html".format(descriptor)
        with open(html_file, "w") as f:
            f.write(html_string)
        return html_string