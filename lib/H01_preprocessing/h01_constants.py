import numpy as np

EXC = [11, 21, 31, 41, 51, 61, 71]
INH = [12, 22, 32, 42, 52, 62, 72]
OTHER = [13, 23, 33, 43, 53, 63, 73]
UNKNOWN = [-1]

UNK_EXC_INH_OTHER = [-1, 1, 2, 3]
EXC_INH_OTHER_UNK = [1, 2, 3, -1]

AXON = [1]
DEND = [2]
AIS = [3]
SOMA = [4]

CELLTYPE_LABELS = {
    -1 : "UNKNOWN",
    1 : "EXC",
    2 : "INH",
    3 : "OTHER",
    11 : "L1-EXC",
    12 : "L1-INH",
    13 : "L1-OTHER",
    21 : "L2-EXC",
    22 : "L2-INH",
    23 : "L2-OTHER",
    31 : "L3-EXC",
    32 : "L3-INH",
    33 : "L3-OTHER",
    41 : "L4-EXC",
    42 : "L4-INH",
    43 : "L4-OTHER",
    51 : "L5-EXC",
    52 : "L5-INH",
    53 : "L5-OTHER",
    61 : "L6-EXC",
    62 : "L6-INH",
    63 : "L6-OTHER",
    71 : "WM-EXC",
    72 : "WM-INH",
    73 : "WM-OTHER",
}

COMPARTMENT_LABELS = { 
    1 : "AXON",
    2 : "DEND",
    3 : "AIS",
    4 : "SOMA"
}

GRID_SIZES = [
        [10000, 10000, 10000],
        [25000, 25000, 25000],
        [50000, 50000, 50000],
        [75000, 75000, 50000],
        [100000, 100000, 50000],
        [100000, 100000, 87500],
        [100000, 100000, 100000],
        [125000, 125000, 125000],
        [150000, 150000, 150000],
        [200000, 200000, 200000]       
    ]

def getLowerBounds():
    return np.array([64651, 39350, 63])


def getLengthConversionFactors():
    return np.array([0.008, 0.008, 0.033])


def getLengthConversionFactorsNanometer():
    return np.array([8, 8, 33])


def getSubvolume(name):
    if name == "testvolume":
        boxMin = np.array([2000, 550, 10])
        boxMax = np.array([2150, 700, 160])
        return boxMin, boxMax
    elif(name == "cellular-volume" or name == "pyramidal-with-layer"):
        # [2.85016e+02 4.01640e+01 5.00000e-02] [3050.348 1909.024  170.478]
        boxMin = np.array([250, 0, 0])
        boxMax = np.array([3100, 1950, 200])
        #boxMin = np.array([300, 50, 10])
        #boxMax = np.array([3050, 1900, 160])
        return boxMin, boxMax
    elif(name == "complete-volume"):
        # [517400. 315416.   2079.] [3867248. 2369408.  172590.]
        boxMin = np.array([0, 0, 0])
        boxMax = np.array([3900000, 2400000, 175000])
        return boxMin, boxMax
    else:
        raise ValueError(name)


def getRadialDistanceFromCenterPoint(position):
    centerPoint = np.array([800*1000, 2650*1000])
    return int(np.linalg.norm(position[0:2]-centerPoint))


def getCorticalAxisAndZRange():
    p0 = np.array([3500000, 500000])
    p1 = np.array([1000000, 2250000])
    zRange = np.array([25000, 150000])
    return p0, p1, zRange


def getLayerDepths():
    return {
        "L4": [-1614122, -1291894]
    }


def getExcInhValues():
    return [2, 1]


def getLayers():
    return ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]


def getCelltypes():
    return ["PYR", "INTER", "OTHER"]


def getRuleProbs(ruleName):
    ruleProbs = None

    ids_excitatory = set(np.loadtxt("/vis/scratchN/bzfharth/H01/meta/ids_excitatory").astype(int))
    ids_inhibitory = set(np.loadtxt("/vis/scratchN/bzfharth/H01/meta/ids_interneuron").astype(int))

    if(ruleName == "exc-exc-attraction"):
        ruleProbs = {
            "preIds": ids_excitatory,
            "postIds": ids_excitatory,
            "beta": 0.8
        }
    elif(ruleName == "exc-inh-attraction"):
        ruleProbs = {
            "preIds": ids_excitatory,
            "postIds": ids_inhibitory,
            "beta": 0.8
        }
    elif(ruleName == "inh-inh-attraction"):
        ruleProbs = {
            "preIds": ids_inhibitory,
            "postIds": ids_inhibitory,
            "beta": 0.8
        }
    elif(ruleName == "inh-exc-attraction"):
        ruleProbs = {
            "preIds": ids_inhibitory,
            "postIds": ids_excitatory,
            "beta": 0.8
        }
    else:
        raise ValueError
    return ruleProbs


def getLabelId(label):
    if(label == "AXON"):
        return 1
    elif(label == "DENDRITE"):
        return 2
    elif(label == "AIS"):
        return 3
    elif(label == "SOMA"):
        return 4
    elif(label == "CILIA"):
        return 5
    elif(label == "UNKNOWN"):
        return 6
    else:
        raise ValueError(label)


def getPyramidalCelltypes():
    celltypes = []
    for layer in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        celltypes.append(getCelltypeId(layer, "PYR"))
    return celltypes


def getInterneuronCelltypes():
    celltypes = []
    for layer in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        celltypes.append(getCelltypeId(layer, "INTER"))
    return celltypes


def getOtherCelltypes():
    celltypes = []
    for layer in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        celltypes.append(getCelltypeId(layer, "OTHER"))
    return celltypes


def getCelltypeId(layer, celltype):
    layerIdx = getLayers().index(layer)
    celltypeIdx = getCelltypes().index(celltype)
    return (layerIdx + 1) * 10 + celltypeIdx + 1


def getCelltypeFromId(celltypeId):
    idx = (celltypeId % 10) - 1
    return getCelltypes()[idx]


def getCompartmentLabels():
    return {
        "axon": 0,
        "axon_exc": 1,
        "axon_inh": 2,
        "axon_initial_segment": 3,
        "dendrite": 10,
        "dendrite_exc_apical_spine": 20,
        "dendrite_exc_basal_spine": 21,
        "dendrite_exc_apical_surface": 22,
        "dendrite_exc_basal_surface": 23,
        "soma": 30,
        "soma_exc_surface": 31,
        "surface_exc-inh": 40,
        "surface_inh-inh": 41,
    }


def getDefaultTargetSpecificity():
    return {
        "axon": ["dendrite", "soma", "axon_initial_segment"],
        "axon_exc": ["dendrite_exc_apical_spine", "dendrite_exc_basal_spine", "surface_exc-inh"],
        "axon_inh": ["dendrite_exc_apical_surface", "dendrite_exc_basal_surface", "soma_exc_surface", "surface_inh-inh"]
    }
