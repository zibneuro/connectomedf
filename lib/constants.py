import numpy as np

# selections
PRESYNAPTIC = "pre"
POSTSYNAPTIC = "post"
PRESYNAPTIC = "pre"
BOTH_SIDES = "pre_post"
VALUE_FILTER = 0
COMBINED_FILTER = 1

# matrix observables
CONPROB_MATRIX = "conprobmatrix"
SYNCOUNT_MATRIX = "syncountmatrix"
SPECIFICITY_MATRIX = "specificitymatrix"
LOCAL_MATRIX = "localmatrix"

# specificity group levels
SPECIFICITY_COMBINED = "specificity-combined"
SPECIFICITY_POPULATION = "specificity-population"
SPECIFICITY_CELLULAR = "specificity-cellular"
SPECIFICITY_SUBCELLULAR = "specificity-subcellular"

# inference on network statistics
INVALID_VALUE = np.nan
POSSIBLE_VALUE = 0.3

# observable names in plots
MOTIFS = "tripletmotifs"
SYNCOUNT_POPULATION = "synapsecounts_population"
SYNCOUNT_SUBCELLULAR = "synapsecounts_subcellular"
SYNCOUNT_EXC_SUBCELLULAR = "synapsecounts_EXC_subcellular"
SYNCOUNT_INH_SUBCELLULAR = "synapsecounts_INH_subcellular"
SYNCLUSTERS = "synapseclusters"
SYNCLUSTERS_X = "synapses per neuron pair"
SYNCLUSTERS_Y = "occurrences in connectome"

# model names in plots 
STR_EMPIRICAL = "empirical"
STR_NULL = r"$H_0$"
STR_P = r"$\alpha_P$"
STR_Pa = r"$\alpha_{Pa}$"
STR_P_disaggregated = r"$\alpha_{Pd}$"
STR_C = r"$\alpha_C$"
STR_S = r"$\alpha_S$"
STR_PC = r"$\alpha_{PC}$"
STR_CP = r"$\alpha_{PC}$"
STR_PS = r"$\alpha_{PS}$"
STR_PSa = r"$\alpha_{PSa}$"
STR_PS_disaggregated = r"$\alpha_{PSd}$"
STR_SP = r"$\alpha_{SP}$"
STR_CS = r"$\alpha_{CS}$"
STR_SC = r"$\alpha_{SC}$"
STR_PCS = r"$\alpha_{PCS}$"
STR_PSC = r"$\alpha_{PSC}$"
STR_PSCa = r"$\alpha_{PSCa}$"
STR_PSCb = r"$\alpha_{PSCb}$"
STR_EMPIRICAL_POTENTIAL = "empirical-potential"


def get_formatted_model_name(model_descriptor):
    mapped_names = {
        'model-null' : STR_NULL,
        'model-P' : STR_P,
        'model-P_disaggregated' : STR_P_disaggregated,
        'model-Pa' : STR_Pa,
        'model-PS' : STR_PS,
        'model-PS_disaggregated' : STR_PS_disaggregated,
        'model-PSa' : STR_PSa,
        'model-PSCa' : STR_PSCa,
        'model-PSCb' : STR_PSCb,
        'model-PSC' : STR_PSC
    }
    if(model_descriptor in mapped_names):
        return mapped_names[model_descriptor]
    else:
        return model_descriptor


STR_SYNAPSE_COUNTS = {
    "ALL_ALL" : "XX",
    "ALL_DEND" : r"$XX_D$",
    "ALL_SOMA" : r"$XX_S$",
    "ALL_AIS" : r"$XX_A$",
    "EXC_EXC" : "EE",
    "EXC_INH" : "EI",
    "EXC_OTHER" : "EO",
    "EXC_UNKNOWN" : "EU",
    "EXC_DEND" : r"$EX_D$",
    "EXC_SOMA" : r"$EX_S$",
    "EXC_AIS" : r"$EX_A$",
    "INH_EXC" : "IE",
    "INH_INH" : "II",  
    "INH_OTHER" : "IO",  
    "INH_UNKNOWN" : "IU",
    "INH_DEND" : r"$IX_D$",
    "INH_SOMA" : r"$IX_S$",
    "INH_AIS" : r"$IX_A$",
    "OTHER_EXC" : "OE",
    "OTHER_INH" : "OI",
    "OTHER_OTHER" : "OO",
    "OTHER_UNKNOWN" : "OU",
    "UNKNOWN_EXC" : "UE",
    "UNKNOWN_INH" : "UI",
    "UNKNOWN_OTHER" : "UO",
    "UNKNOWN_UNKNOWN" : "UU",
    
    "EXC_EXC-DEND" : r"$EE_D$",
    "EXC_EXC-SOMA" : r"$EE_S$",
    "EXC_EXC-AIS" : r"$EE_A$",
    "EXC_INH-DEND" : r"$EI_D$",
    "EXC_INH-SOMA" : r"$EI_S$",
    "EXC_INH-AIS" : r"$EI_A$",
    "EXC_OTHER-DEND" : r"$EO_D$",
    "EXC_OTHER-SOMA" : r"$EO_S$",
    "EXC_OTHER-AIS" : r"$EO_A$",
    "EXC_UNKNOWN-DEND" : r"$EU_D$",
    "EXC_UNKNOWN-SOMA" : r"$EU_S$",
    "EXC_UNKNOWN-AIS" : r"$EU_A$",
    
    "INH_EXC-DEND" : r"$IE_D$",
    "INH_EXC-SOMA" : r"$IE_S$",
    "INH_EXC-AIS" : r"$IE_A$",
    "INH_INH-DEND" : r"$II_D$",
    "INH_INH-SOMA" : r"$II_S$",
    "INH_INH-AIS" : r"$II_A$",
    "INH_OTHER-DEND" : r"$IO_D$",
    "INH_OTHER-SOMA" : r"$IO_S$",
    "INH_OTHER-AIS" : r"$IO_A$",
    "INH_UNKNOWN-DEND" : r"$IU_D$",
    "INH_UNKNOWN-SOMA" : r"$IU_S$",
    "INH_UNKNOWN-AIS" : r"$IU_A$",

    "OTHER_EXC-DEND" : r"$OE_D$",
    "OTHER_EXC-SOMA" : r"$OE_S$",
    "OTHER_EXC-AIS" : r"$OE_A$",
    "OTHER_INH-DEND" : r"$OI_D$",
    "OTHER_INH-SOMA" : r"$OI_S$",
    "OTHER_INH-AIS" : r"$OI_A$",
    "OTHER_OTHER-DEND" : r"$OO_D$",
    "OTHER_OTHER-SOMA" : r"$OO_S$",
    "OTHER_OTHER-AIS" : r"$OO_A$",
    "OTHER_UNKNOWN-DEND" : r"$OU_D$",
    "OTHER_UNKNOWN-SOMA" : r"$OU_S$",
    "OTHER_UNKNOWN-AIS" : r"$OU_A$",
    
    "UNKNOWN_EXC-DEND" : r"$UE_D$",
    "UNKNOWN_EXC-SOMA" : r"$UE_S$",
    "UNKNOWN_EXC-AIS" : r"$UE_A$",
    "UNKNOWN_INH-DEND" : r"$UI_D$",
    "UNKNOWN_INH-SOMA" : r"$UI_S$",
    "UNKNOWN_INH-AIS" : r"$UI_A$",
    "UNKNOWN_OTHER-DEND" : r"$UO_D$",
    "UNKNOWN_OTHER-SOMA" : r"$UO_S$",
    "UNKNOWN_OTHER-AIS" : r"$UO_A$",
    "UNKNOWN_UNKNOWN-DEND" : r"$UU_D$",
    "UNKNOWN_UNKNOWN-SOMA" : r"$UU_S$",
    "UNKNOWN_UNKNOWN-AIS" : r"$UU_A$"
}

def get_short_descriptor(descriptor):
    assert "_" in descriptor
    pre, post = descriptor.split("_")

    key_words = {
        "EXC" : "E",
        "INH" : "I",
        "OTHER" : "O",
        "UNKNOWN" : "U",
        "DEND" : "D",
        "SOMA" : "S",
        "AIS" : "A",
        "ALL" : "X",
        "20" : "unk",
        "21" : "bip",
        "22" : "Bas",
        "23" : "Cha",
        "24" : "Mar",
        "25" : "Neu"
    }

    def get_key(part):
        if part in key_words:
            return key_words[part]
        else:
            return part
        
    def get_subsuperscript(part):
        if(part.isdigit()):
            return r"^{{{}}}".format(part)
        else:
            return "_" + part

    def get_short_str(word):
        parts = word.split("-") 
        short_str = get_key(parts[0])
        
        if(len(parts) == 1):   
            pass
        elif(len(parts) == 2):
            short_str += r"_{{{}}}".format(get_key(parts[1]))
        elif(len(parts) == 3):
            short_str += r"_{{{}}}".format(get_key(parts[1]) + "," + get_key(parts[2]))
        else:
            raise ValueError(f"Invalid descriptor {word}")
        
        return short_str
    
    combined = get_short_str(pre) + get_short_str(post)
    return r"${{{}}}$".format(combined)


# model names
EMPIRICAL = "empirical"
MODEL_NULL = "model-null"
MODEL_P = "model-P"
MODEL_C = "model-C"
MODEL_S = "model-S"
MODEL_PC = "model-PC"
MODEL_CP = "model-CP"
MODEL_PS = "model-PS"
MODEL_SP = "model-PS"
MODEL_CS = "model-CS"
MODEL_SC = "model-SC"
MODEL_PCS = "model-PCS"
MODEL_PSC = "model-PSC"
MODEL_CPS = "model-CPS"
MODEL_CSP = "model-CSP"
MODEL_SPC = "model-SPC"
MODEL_SCP = "model-SCP"

MODEL_P_nonsequential = "model-P-nonsequential"
MODEL_PS_nonsequential = "model-PS-nonsequential"
MODEL_Ca_nonsequential = "model-Ca-nonsequential"
MODEL_Cb_nonsequential = "model-Cb-nonsequential"
MODEL_PCa_nonsequential = "model-PCa-nonsequential"
MODEL_PCb_nonsequential = "model-PCb-nonsequential"
MODEL_PCaS_nonsequential = "model-PCaS-nonsequential"
MODEL_PCbS_nonsequential = "model-PCbS-nonsequential"
MODEL_PCS_nonsequential = "model-PCS-nonsequential"

MODEL_Ca = "model-Ca"
MODEL_Cb = "model-Cb"
MODEL_PSCa = "model-PSCa"
MODEL_PSCb = "model-PSCb"

MODEL_Pa = "model-Pa"
MODEL_PaS = "model-PaS"
MODEL_PSa = "model-PSa"
MODEL_PaSb = "model-PaSb"


MODEL_P_disaggregated = "model-P_disaggregated"
MODEL_PS_disaggregated = "model-PS_disaggregated"

MODEL_S_EXC = "model-S-EXC"
MODEL_S_INH = "model-S-INH"

MODEL_REFERENCE = "model_reference"
MODEL_CURRENT = "model_current"

# SELECTIONS
SELECTION_CELLTYPE = ["EXC_EXC", "EXC_INH", "INH_EXC", "INH_INH"]
SELECTION_CELLTYPE_H01_ALL = ["EXC_EXC", "EXC_INH", "EXC_OTHER", "EXC_UNKNOWN", \
                              "INH_EXC", "INH_INH", "INH_OTHER", "INH_UNKNOWN", \
                              "OTHER_EXC", "OTHER_INH", "OTHER_OTHER", "OTHER_UNKNOWN", \
                              "UNKNOWN_EXC", "UNKNOWN_INH", "UNKNOWN_OTHER", "UNKNOWN_UNKNOWN"]

CELLTYPE_LABELS = {
    -1 : "UNKNOWN",
    1 : "EXC",
    2 : "INH",
    20 : "INH-20",
    21 : "INH-21",
    22 : "INH-22",
    23 : "INH-23",
    24 : "INH-24",
    25 : "INH-25",
}

COMPARTMENT_LABELS = {
    -1 : "UNKNOWN",
    1 : "AXON",
    2 : "DEND",
    3 : "AIS",
    4 : "SOMA"
}


SELECTION_CELLTYPE_VIS_ALL = ["EXC_EXC", "EXC_INH", "EXC_UNKNOWN", \
                              "INH_EXC", "INH_INH", "INH_UNKNOWN", \
                              "UNKNOWN_EXC", "UNKNOWN_INH", "UNKNOWN_UNKNOWN"]

SELECTION_VIS_EXC_INH_SUBPOPULATIONS = [
    "EXC_INH-20-SOMA", "EXC_INH-20-DEND", "EXC_INH-20-AIS", "EXC_INH-20-UNKNOWN",
    "EXC_INH-21-SOMA", "EXC_INH-21-DEND", "EXC_INH-21-AIS", "EXC_INH-21-UNKNOWN",
    "EXC_INH-22-SOMA", "EXC_INH-22-DEND", "EXC_INH-22-AIS", "EXC_INH-22-UNKNOWN",
    "EXC_INH-23-SOMA", "EXC_INH-23-DEND", "EXC_INH-23-AIS", "EXC_INH-23-UNKNOWN",
    "EXC_INH-24-SOMA", "EXC_INH-24-DEND", "EXC_INH-24-AIS", "EXC_INH-24-UNKNOWN",
    "EXC_INH-25-SOMA", "EXC_INH-25-DEND", "EXC_INH-25-AIS", "EXC_INH-25-UNKNOWN",
]

SELECTION_VIS_EXC_INH_SUBPOPULATIONS_SD = [
    "EXC_INH-20-SOMA", "EXC_INH-20-DEND",
    "EXC_INH-21-SOMA", "EXC_INH-21-DEND",
    "EXC_INH-22-SOMA", "EXC_INH-22-DEND",
    "EXC_INH-23-SOMA", "EXC_INH-23-DEND",
    "EXC_INH-24-SOMA", "EXC_INH-24-DEND",
    "EXC_INH-25-SOMA", "EXC_INH-25-DEND",
]

SELECTION_VIS_INH_SUBPOPULATIONS = [
    "INH-20_EXC-SOMA", "INH-20_EXC-DEND", "INH-20_EXC-AIS", "INH-20_EXC-UNKNOWN", "INH-20_INH-SOMA", "INH-20_INH-DEND", "INH-20_INH-AIS", "INH-20_INH-UNKNOWN",
    "INH-21_EXC-SOMA", "INH-21_EXC-DEND", "INH-21_EXC-AIS", "INH-21_EXC-UNKNOWN", "INH-21_INH-SOMA", "INH-21_INH-DEND", "INH-21_INH-AIS", "INH-21_INH-UNKNOWN",
    "INH-22_EXC-SOMA", "INH-22_EXC-DEND", "INH-22_EXC-AIS", "INH-22_EXC-UNKNOWN", "INH-22_INH-SOMA", "INH-22_INH-DEND", "INH-22_INH-AIS", "INH-22_INH-UNKNOWN",
    "INH-23_EXC-SOMA", "INH-23_EXC-DEND", "INH-23_EXC-AIS", "INH-23_EXC-UNKNOWN", "INH-23_INH-SOMA", "INH-23_INH-DEND", "INH-23_INH-AIS", "INH-23_INH-UNKNOWN",
    "INH-24_EXC-SOMA", "INH-24_EXC-DEND", "INH-24_EXC-AIS", "INH-24_EXC-UNKNOWN", "INH-24_INH-SOMA", "INH-24_INH-DEND", "INH-24_INH-AIS", "INH-24_INH-UNKNOWN",
    "INH-25_EXC-SOMA", "INH-25_EXC-DEND", "INH-25_EXC-AIS", "INH-25_EXC-UNKNOWN", "INH-25_INH-SOMA", "INH-25_INH-DEND", "INH-25_INH-AIS", "INH-25_INH-UNKNOWN",
]

SELECTION_VIS_INH_SUBPOPULATIONS_EXC = [
    "INH-20_EXC-SOMA", "INH-20_EXC-DEND", "INH-20_EXC-AIS",
    "INH-21_EXC-SOMA", "INH-21_EXC-DEND", "INH-21_EXC-AIS",
    "INH-22_EXC-SOMA", "INH-22_EXC-DEND", "INH-22_EXC-AIS",
    "INH-23_EXC-SOMA", "INH-23_EXC-DEND", "INH-23_EXC-AIS",
    "INH-24_EXC-SOMA", "INH-24_EXC-DEND", "INH-24_EXC-AIS",
    "INH-25_EXC-SOMA", "INH-25_EXC-DEND", "INH-25_EXC-AIS",
]

SELECTION_VIS_INH_SUBPOPULATIONS_INH = [
    "INH-20_INH-SOMA", "INH-20_INH-DEND", "INH-20_INH-AIS", 
    "INH-21_INH-SOMA", "INH-21_INH-DEND", "INH-21_INH-AIS", 
    "INH-22_INH-SOMA", "INH-22_INH-DEND", "INH-22_INH-AIS", 
    "INH-23_INH-SOMA", "INH-23_INH-DEND", "INH-23_INH-AIS", 
    "INH-24_INH-SOMA", "INH-24_INH-DEND", "INH-24_INH-AIS", 
    "INH-25_INH-SOMA", "INH-25_INH-DEND", "INH-25_INH-AIS", 
]

SELECTION_COMPARTMENT = ["ALL_SOMA", "ALL_DEND", "ALL_AIS"]
SELECTION_PRECT_COMPARTMENT = ["EXC_SOMA", "EXC_DEND", "EXC_AIS", "INH_SOMA", "INH_DEND", "INH_AIS"]
SELECTION_EXC_COMPARTMENT = ["ALL_EXC-SOMA", "ALL_EXC-DEND", "ALL_EXC-AIS"]
SELECTION_INH_COMPARTMENT = ["ALL_INH-SOMA", "ALL_INH-DEND", "ALL_INH-AIS"]

SELECTION_EXC_SUBCELLULAR = ["EXC_EXC-SOMA", "EXC_EXC-DEND", "EXC_EXC-AIS", "EXC_INH-SOMA", "EXC_INH-DEND", "EXC_INH-AIS"]
SELECTION_INH_SUBCELLULAR = ["INH_EXC-SOMA", "INH_EXC-DEND", "INH_EXC-AIS", "INH_INH-SOMA", "INH_INH-DEND", "INH_INH-AIS"]

SELECTION_EMPIRICAL_NULL_SELECTED = ["EXC_EXC", "EXC_INH", "INH_EXC", "INH_INH", "EXC_INH-SOMA", "INH_EXC-AIS"]

SELECTION_EXC_COMPARTMENT_H01_ALL = ["EXC_EXC-SOMA", "EXC_EXC-DEND", "EXC_EXC-AIS", \
                                     "EXC_INH-SOMA", "EXC_INH-DEND", "EXC_INH-AIS", \
                                     "EXC_OTHER-SOMA", "EXC_OTHER-DEND", "EXC_OTHER-AIS", \
                                     "EXC_UNKNOWN-SOMA", "EXC_UNKNOWN-DEND", "EXC_UNKNOWN-AIS"]

SELECTION_INH_COMPARTMENT_H01_ALL = ["INH_EXC-SOMA", "INH_EXC-DEND", "INH_EXC-AIS", \
                                     "INH_INH-SOMA", "INH_INH-DEND", "INH_INH-AIS", \
                                     "INH_OTHER-SOMA", "INH_OTHER-DEND", "INH_OTHER-AIS", \
                                     "INH_UNKNOWN-SOMA", "INH_UNKNOWN-DEND", "INH_UNKNOWN-AIS"]

SELECTION_OTHER_COMPARTMENT_H01_ALL = ["OTHER_EXC-SOMA", "OTHER_EXC-DEND", "OTHER_EXC-AIS", \
                                     "OTHER_INH-SOMA", "OTHER_INH-DEND", "OTHER_INH-AIS", \
                                     "OTHER_OTHER-SOMA", "OTHER_OTHER-DEND", "OTHER_OTHER-AIS", \
                                     "OTHER_UNKNOWN-SOMA", "OTHER_UNKNOWN-DEND", "OTHER_UNKNOWN-AIS"]

SELECTION_UNKNOWN_COMPARTMENT_H01_ALL = ["UNKNOWN_EXC-SOMA", "UNKNOWN_EXC-DEND", "UNKNOWN_EXC-AIS", \
                                     "UNKNOWN_INH-SOMA", "UNKNOWN_INH-DEND", "UNKNOWN_INH-AIS", \
                                     "UNKNOWN_OTHER-SOMA", "UNKNOWN_OTHER-DEND", "UNKNOWN_OTHER-AIS", \
                                     "UNKNOWN_UNKNOWN-SOMA", "UNKNOWN_UNKNOWN-DEND", "UNKNOWN_UNKNOWN-AIS"]

SELECTION_MIXED_SUBCELLULAR = ["EXC_EXC-SOMA", "EXC_EXC-DEND", "EXC_EXC-AIS", "INH_INH-SOMA", "INH_INH-DEND"]

SELECTION_MOTIF = [f"MOTIF-{k}" for k in range(1, 17)]

SELECTION_CLUSTER = [f"CLUSTER-{k}" for k in range(0, 21)]
SELECTION_CLUSTER_6 = [f"CLUSTER-{k}" for k in range(0, 7)]
SELECTION_CLUSTER_8 = [f"CLUSTER-{k}" for k in range(0, 9)]








