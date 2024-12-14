import os
import glob
import pandas as pd
import numpy as np

from . import util


class Gridder():
    def __init__(self, gridCacheDir):
        self.eps = 1 # nanometer
        self.gridCacheDir = gridCacheDir        
        os.makedirs(self.gridCacheDir, exist_ok=True)

    def setPositions(self, xyz):
        self.positions = xyz
        self.min = np.min(self.positions, axis=0)
        self.max = np.max(self.positions, axis=0)
        self.positions_zero_based = self.positions - self.min
        self.extent = np.max(self.positions_zero_based, axis=0)    
        self.gridSize = None
        
    def computeGrid(self, gridSize):
        gridSize = np.array(gridSize)
        indices = np.floor_divide(self.positions_zero_based, gridSize)
        base = np.max(indices) + 1
        baseFactors = np.array([base**2,base,1])            
        indicesScalar = np.dot(indices, baseFactors)        

        x_min = self.min[0] + indices[:, 0] * gridSize
        y_min = self.min[1] + indices[:, 1] * gridSize
        z_min = self.min[2] + indices[:, 2] * gridSize
        x_max = x_min + gridSize
        y_max = y_min + gridSize
        z_max = z_min + gridSize
        
        df_grid_meta = pd.DataFrame({
            'overlap_volume' : indicesScalar,
            'min_x': x_min,
            'min_y': y_min,
            'min_z': z_min,
            'max_x': x_max,
            'max_y': y_max,
            'max_z': z_max
        })

        df_grid_meta = df_grid_meta.drop_duplicates(subset="overlap_volume").reset_index(drop=True)

        return indicesScalar, df_grid_meta     