EXC = [1]
INH = [20, 21, 22, 23, 24, 25]
INH_20 = [20]
INH_21 = [21]
INH_22 = [22]
INH_23 = [23]
INH_24 = [24]
INH_25 = [25]
UNKNOWN = [-1]
AXON = [1]
DEND = [2]
AIS = [3]
SOMA = [4]

E = 1
I = 2
EXC_INH = [1, 2]
EXC_INH_UNKNOWN = [1, 2, -1]

CELLTYPES_ALL = [1, 20, 21, 22, 23, 24, 25, -1]

CELLTYPE_LABELS = {
    -1 : "UNKNOWN",
    1 : "EXC",
    2 : "INH",
    20 : "INH-20 (unknown)",
    21 : "INH-21 (bipolar)",
    22 : "INH-22 (Basket)",
    23 : "INH-23 (Chandelier)",
    24 : "INH-24 (Martinotti)",
    25 : "INH-25 (Neurogliaform)",
}

CELLTYPE_LABELS_SHORT = {
    -1 : r"$U$",
    1 : r"$E$",
    2 : r"$I$",
    20 : r"$I_{unk}$",
    21 : r"$I_{bip}$",
    22 : r"$I_{Bas}$",
    23 : r"$I_{Cha}$",
    24 : r"$I_{Mar}$",
    25 : r"$I_{Neu}$",
}

COMPARTMENT_LABELS = {
    -1 : "UNKNOWN",
    1 : "AXON",
    2 : "DEND",
    3 : "AIS",
    4 : "SOMA"
}

GRID_SIZES = [        
        [5000, 5000, 5000],
        [10000, 10000, 10000],
        [15000, 15000, 15000],
        [20000, 20000, 20000],
        [25000, 25000, 25000],
        [30000, 30000, 30000],
        [40000, 40000, 40000],
        [50000, 50000, 50000],
        [60000, 60000, 60000],
        [70000, 70000, 70000]
    ]

def getCelltypeId(celltype):
    if(celltype == "PYR"):
        return 1
    elif(celltype == "INTER-unknown"):
        return 20
    elif(celltype == "INTER-bipolar"):
        return 21
    elif(celltype == "INTER-basket"):
        return 22
    elif(celltype == "INTER-chandelier"):
        return 23
    elif(celltype == "INTER-martinotti"):
        return 24
    elif(celltype == "INTER-neurogliaform"):
        return 25    
    else:
        raise ValueError(celltype)


def getExcitatoryCelltypeIds():
    return [1]

def getInhibitoryCelltypeIds():
    return [20, 21, 22, 23, 24, 25]


def getCompartmentId(label):
    if(label == "AXON"):
        return 1
    elif(label == "DENDRITE"):
        return 2
    elif(label == "AIS"):
        return 3
    elif(label == "SOMA"):
        return 4    
    else:
        raise ValueError(label)
