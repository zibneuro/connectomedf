import numpy as np

GRIDSIZE = np.array([50, 50, 50])

def setGridSize(gridDescriptor):
    steps = gridDescriptor.split("-")
    for i in range(0, 3):
        value = steps[i]
        if(not float(value).is_integer()):
            raise RuntimeError("invalid grid step size {}".format(steps[i]))
        valueInt = int(value)
        GRIDSIZE[i] = valueInt


def getShiftedGridBounds(boxMin, boxMax):
    ixiyiz_min = np.array([0,0,0], dtype=int)
    boxSize = boxMax - boxMin
    I, R = np.divmod(boxSize, GRIDSIZE)    
    ixiyiz_max = I.astype(int)

    if(np.count_nonzero(R) != 0):
        raise RuntimeError("box dimensions incompatible with grid size")

    ixiyiz_delta = np.array([ixiyiz_max[0]-ixiyiz_min[0], ixiyiz_max[1]-ixiyiz_min[1], ixiyiz_max[2]-ixiyiz_min[2]])
    numCells = ixiyiz_delta[0] * ixiyiz_delta[1] * ixiyiz_delta[2]
    return {
        "boxMin" : boxMin,
        "boxMax" : boxMax,
        "ixiyiz_min" : ixiyiz_min,
        "ixiyiz_max" : ixiyiz_max,
        "ixiyiz_delta" : ixiyiz_delta,
        "numCells" : numCells
    }


def correctBoundaryPoints(ixiyiz, ixiyiz_max):
    for i in range(0,3):
        idxBoundaryCase = ixiyiz[:,i] == ixiyiz_max[i]
        ixiyiz[idxBoundaryCase,i] = ixiyiz_max[i] - 1


def getArrayIndex(gridBounds, cube):    
    origin = gridBounds["ixiyiz_min"]
    gridRange = gridBounds["ixiyiz_delta"]
    nx = gridRange[0]
    ny = gridRange[1]
    ix = cube[0] - origin[0]
    iy = cube[1] - origin[1]
    iz = cube[2] - origin[2]
    if(ix < 0 or iy < 0 or iz < 0):
        raise RuntimeError("negative rel. index")
    idx = iz * nx * ny + iy * nx + ix
    if(idx >= gridBounds["numCells"]):
        raise RuntimeError("idx exceeds bounds")
    return idx
    

def getCubeFromArrayIndex(gridBounds, arrayIndex):
    origin = gridBounds["ixiyiz_min"]
    gridRange = gridBounds["ixiyiz_delta"]
    nx = gridRange[0]
    ny = gridRange[1]
    iz, rxy = np.divmod(arrayIndex, (nx * ny))
    iy, ix = np.divmod(rxy, nx)
    cx = int(ix + origin[0])
    cy = int(iy + origin[1])
    cz = int(iz + origin[2])
    return (cx, cy, cz)


def getCubeIdsFromShiftedGridBounds(gridBounds, positions):    
    positionsShiftedBoxOrigin = positions - gridBounds["boxMin"]
    ixiyiz = np.floor_divide(positionsShiftedBoxOrigin, GRIDSIZE)   

    ixiyiz_max = gridBounds["ixiyiz_max"]    
    correctBoundaryPoints(ixiyiz, ixiyiz_max)
    
    if(np.any(ixiyiz < np.zeros(3,dtype=int)) or np.any(ixiyiz >= ixiyiz_max)):
        raise RuntimeError("cube indices out of bounds")

    n = ixiyiz.shape[0]
    cubeIds = np.zeros(n, dtype=int)
    for i in range(0,n):
        cubeIds[i] = getArrayIndex(gridBounds, ixiyiz[i])

    return cubeIds


def getCubeOriginForCubeId(gridBounds, cubeId):
    cube_ixiyiz = getCubeFromArrayIndex(gridBounds, cubeId)
    cubeOrigin = gridBounds["boxMin"] + np.multiply(cube_ixiyiz, GRIDSIZE)
    return cubeOrigin 