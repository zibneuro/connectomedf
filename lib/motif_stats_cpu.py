from .observable_matrix import compute_probability_matrix
from .eval_motifs import getMotifMasks, calcMotifProbability, aggregateProbabilties_16, getDeviations

import numpy as np
import math


class MotifStatisticsCPU:

    def __init__(self):
        self.masks, self.masks_inv = getMotifMasks(3)


    def get_motif_distribution(self, synapse_matrix, discrete):
            n_rows, n_cols = synapse_matrix.shape
            assert n_rows == n_cols
            num_neurons = n_rows

            prob_matrix = compute_probability_matrix(synapse_matrix, discrete)

            num_triplet_samples = math.comb(num_neurons, 3)
            p_model = np.zeros((num_triplet_samples, 6))
            idx = 0
            for i in range(0, num_neurons):
                for j in range(i+1, num_neurons):
                    for k in range(j+1, num_neurons):
                        p_model[idx, 0] = prob_matrix[i, j]
                        p_model[idx, 1] = prob_matrix[j, i]
                        p_model[idx, 2] = prob_matrix[i, k]
                        p_model[idx, 3] = prob_matrix[k, i]
                        p_model[idx, 4] = prob_matrix[j, k]
                        p_model[idx, 5] = prob_matrix[k, j]
                        idx += 1

            # time-consuming section 
            # --------------------------------------------------------
            p_average_all_pairs = np.mean(prob_matrix)

            p_model_inv = 1 - p_model
            p_avg = p_average_all_pairs * np.ones(6)
            p_avg = p_avg.reshape((-1, 6))
            p_avg_inv = 1 - p_avg
                
            motif_probabilities_64_random = {}
            motif_probabilities_64_model = {}
            for i in range(0, self.masks.shape[0]):
                mask = self.masks[i, :]
                mask_inv = self.masks_inv[i, :]
                motifKey = tuple(mask.astype(int))
                motif_probabilities_64_random[motifKey] = calcMotifProbability(p_avg, p_avg_inv, mask, mask_inv)
                motif_probabilities_64_model[motifKey] = calcMotifProbability(p_model, p_model_inv, mask, mask_inv)
            
            # --------------------------------------------------------

            motif_probabilities_16_random = aggregateProbabilties_16(motif_probabilities_64_random)
            motif_probabilities_16_model = aggregateProbabilties_16(motif_probabilities_64_model)
            
            deviations = getDeviations(motif_probabilities_16_random, motif_probabilities_16_model, idx_offset=1)
            return deviations