import os
import sys

import h01_util

def loadIdsFromExportedFile(filename):
    ids = set()
    with open(filename) as f:
        lines = f.readlines()
        if(len(lines) != 1):
            raise RuntimeError
        parts = lines[0].split(",")
        for part in parts:
            ids.add(int(part.strip()))
    return ids

if __name__ == "__main__":
    metaFolder = os.path.join(h01_util.getBaseFolder(), "meta")
    selectionsFolder = os.path.join(metaFolder, "neuroglancer_selections")

    selected_L1 = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_L1"))
    print("L1", len(selected_L1))
    selected_L2 = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_L2"))
    print("L2", len(selected_L2))
    selected_L3 = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_L3"))
    print("L3", len(selected_L3))
    selected_L4 = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_L4"))
    print("L4", len(selected_L4))
    selected_L5 = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_L5"))
    print("L5", len(selected_L5))
    selected_L6 = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_L6"))
    print("L6", len(selected_L6))
    selected_WM = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_WM"))
    print("WM", len(selected_WM))

    selected_pyramidal = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_pyramidal"))
    print("pyramidal", len(selected_pyramidal))
    selected_interneuron = loadIdsFromExportedFile(os.path.join(selectionsFolder, "classified_interneuron"))
    print("interneuron", len(selected_interneuron))
    assert len(selected_pyramidal & selected_interneuron) == 0

    ids_classified = selected_L1 | selected_L2 | selected_L3 | selected_L4 | selected_L5 | selected_L6 | selected_WM
    print("classified", len(ids_classified))
    h01_util.writeIds(os.path.join(metaFolder, "ids_classified"), ids_classified)
    lengthSelections = len(selected_L1) + len(selected_L2) + len(selected_L3) + len(selected_L4) + len(selected_L5) + len(selected_L6) + len(selected_WM)
    assert len(ids_classified) == lengthSelections

    ids_other = ids_classified - (selected_pyramidal | selected_interneuron)
    print("other celltype", len(ids_other))

    ids_L1_PYR = selected_L1 & selected_pyramidal
    ids_L1_INTER = selected_L1 & selected_interneuron
    ids_L1_OTHER = selected_L1 & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_L1_PYR"), ids_L1_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L1_INTER"), ids_L1_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L1_OTHER"), ids_L1_OTHER)

    ids_L2_PYR = selected_L2 & selected_pyramidal
    ids_L2_INTER = selected_L2 & selected_interneuron
    ids_L2_OTHER = selected_L2 & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_L2_PYR"), ids_L2_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L2_INTER"), ids_L2_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L2_OTHER"), ids_L2_OTHER)

    ids_L3_PYR = selected_L3 & selected_pyramidal
    ids_L3_INTER = selected_L3 & selected_interneuron
    ids_L3_OTHER = selected_L3 & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_L3_PYR"), ids_L3_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L3_INTER"), ids_L3_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L3_OTHER"), ids_L3_OTHER)

    ids_L4_PYR = selected_L4 & selected_pyramidal
    ids_L4_INTER = selected_L4 & selected_interneuron
    ids_L4_OTHER = selected_L4 & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_L4_PYR"), ids_L4_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L4_INTER"), ids_L4_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L4_OTHER"), ids_L4_OTHER)

    ids_L5_PYR = selected_L5 & selected_pyramidal
    ids_L5_INTER = selected_L5 & selected_interneuron
    ids_L5_OTHER = selected_L5 & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_L5_PYR"), ids_L5_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L5_INTER"), ids_L5_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L5_OTHER"), ids_L5_OTHER)

    ids_L6_PYR = selected_L6 & selected_pyramidal
    ids_L6_INTER = selected_L6 & selected_interneuron
    ids_L6_OTHER = selected_L6 & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_L6_PYR"), ids_L6_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L6_INTER"), ids_L6_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_L6_OTHER"), ids_L6_OTHER)

    ids_WM_PYR = selected_WM & selected_pyramidal
    ids_WM_INTER = selected_WM & selected_interneuron
    ids_WM_OTHER = selected_WM & ids_other
    h01_util.writeIds(os.path.join(metaFolder, "ids_WM_PYR"), ids_WM_PYR)
    h01_util.writeIds(os.path.join(metaFolder, "ids_WM_INTER"), ids_WM_INTER)
    h01_util.writeIds(os.path.join(metaFolder, "ids_WM_OTHER"), ids_WM_OTHER)

    allIds = set()
    for layer in ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]:
        for celltype in ["PYR", "INTER", "OTHER"]:
            idsFromFile = h01_util.loadIds(os.path.join(metaFolder, "ids_{}_{}".format(layer, celltype)))
            assert len(idsFromFile & allIds) == 0
            allIds |= idsFromFile
    assert allIds == ids_classified
