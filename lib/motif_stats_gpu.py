from .observable_matrix import compute_probability_matrix
from .eval_motifs import getMotifMasks, calcMotifProbability, aggregateProbabilties_16, getDeviations

import numpy as np
import cupy as cp
import math


class MotifStatisticsGPU:

    def __init__(self, num_neurons, init_blocks = False, flat_probabilties = False):
        self.masks, self.masks_inv = getMotifMasks(3)
        self.num_neurons = num_neurons
        self.flat_probabilties = flat_probabilties

        if(flat_probabilties):
            self.init_flat_kernel()
        else:
            if(init_blocks):
                assert num_neurons % 3 == 0
                self.triplet_probability_indices = self.get_disjoint_triplet_blocks()       
            else:
                self.triplet_probability_indices = self.get_nonredundant_triplets()

            self.I = cp.array(self.triplet_probability_indices.astype(np.uint32), dtype="uint32")
            self.init_kernel()

    def get_disjoint_triplet_blocks(self):
        num_triplet_samples = int(self.num_neurons / 3)
        p_indices = np.zeros((num_triplet_samples, 6), dtype=int)                
        for idx in range(0, num_triplet_samples):
            i = idx * 3
            j = i + 1
            k = i + 2
            p_indices[idx, 0] = i * self.num_neurons + j
            p_indices[idx, 1] = j * self.num_neurons + i
            p_indices[idx, 2] = i * self.num_neurons + k
            p_indices[idx, 3] = k * self.num_neurons + i
            p_indices[idx, 4] = j * self.num_neurons + k
            p_indices[idx, 5] = k * self.num_neurons + j
        return p_indices

    def get_nonredundant_triplets(self):
        num_triplet_samples = math.comb(self.num_neurons, 3)
        p_indices = np.zeros((num_triplet_samples, 6), dtype=int)
        idx = 0
        for i in range(0, self.num_neurons):
            for j in range(i+1, self.num_neurons):
                for k in range(j+1, self.num_neurons):
                    p_indices[idx, 0] = i * self.num_neurons + j
                    p_indices[idx, 1] = j * self.num_neurons + i
                    p_indices[idx, 2] = i * self.num_neurons + k
                    p_indices[idx, 3] = k * self.num_neurons + i
                    p_indices[idx, 4] = j * self.num_neurons + k
                    p_indices[idx, 5] = k * self.num_neurons + j
                    idx += 1                    
        return p_indices


    def init_kernel(self):
        self.calc_motif_probs = cp.ElementwiseKernel(
            'uint32 ab, uint32 ba, uint32 ac, uint32 ca, uint32 bc, uint32 cb, raw float64 p',
            'raw float64 p_motifs',
            """
                double p_ab = p[ab];
                double p_ba = p[ba];
                double p_ac = p[ac];
                double p_ca = p[ca];
                double p_bc = p[bc];
                double p_cb = p[cb];

                double p_ab_inv = 1 - p_ab;
                double p_ba_inv = 1 - p_ba;
                double p_ac_inv = 1 - p_ac;
                double p_ca_inv = 1 - p_ca;
                double p_bc_inv = 1 - p_bc;
                double p_cb_inv = 1 - p_cb;

                atomicAdd(&p_motifs[0], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[1], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[2], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[3], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[4], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[5], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[6], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[7], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[8], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[9], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[10], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[11], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[12], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[13], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[14], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[15], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[16], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[17], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[18], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[19], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[20], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[21], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[22], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[23], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[24], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[25], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[26], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[27], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[28], p_ab_inv * p_ba * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[29], p_ab_inv * p_ba * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[30], p_ab_inv * p_ba * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[31], p_ab_inv * p_ba * p_ac * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[32], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[33], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[34], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[35], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[36], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[37], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[38], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[39], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[40], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[41], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[42], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[43], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[44], p_ab * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[45], p_ab * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[46], p_ab * p_ba_inv * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[47], p_ab * p_ba_inv * p_ac * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[48], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[49], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[50], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[51], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[52], p_ab * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[53], p_ab * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[54], p_ab * p_ba * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[55], p_ab * p_ba * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[56], p_ab * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[57], p_ab * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[58], p_ab * p_ba * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[59], p_ab * p_ba * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[60], p_ab * p_ba * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[61], p_ab * p_ba * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[62], p_ab * p_ba * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[63], p_ab * p_ba * p_ac * p_ca * p_bc * p_cb);
            """,
            'calc_motif_probs'
        )

    def init_flat_kernel(self):
        self.calc_motif_probs_flat = cp.ElementwiseKernel(
            'float64 p_ab, float64 p_ba, float64 p_ac, float64 p_ca, float64 p_bc, float64 p_cb',
            'raw float64 p_motifs',
            """
                double p_ab_inv = 1 - p_ab;
                double p_ba_inv = 1 - p_ba;
                double p_ac_inv = 1 - p_ac;
                double p_ca_inv = 1 - p_ca;
                double p_bc_inv = 1 - p_bc;
                double p_cb_inv = 1 - p_cb;

                atomicAdd(&p_motifs[0], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[1], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[2], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[3], p_ab_inv * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[4], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[5], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[6], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[7], p_ab_inv * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[8], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[9], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[10], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[11], p_ab_inv * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[12], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[13], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[14], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[15], p_ab_inv * p_ba_inv * p_ac * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[16], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[17], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[18], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[19], p_ab_inv * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[20], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[21], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[22], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[23], p_ab_inv * p_ba * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[24], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[25], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[26], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[27], p_ab_inv * p_ba * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[28], p_ab_inv * p_ba * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[29], p_ab_inv * p_ba * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[30], p_ab_inv * p_ba * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[31], p_ab_inv * p_ba * p_ac * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[32], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[33], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[34], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[35], p_ab * p_ba_inv * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[36], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[37], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[38], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[39], p_ab * p_ba_inv * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[40], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[41], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[42], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[43], p_ab * p_ba_inv * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[44], p_ab * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[45], p_ab * p_ba_inv * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[46], p_ab * p_ba_inv * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[47], p_ab * p_ba_inv * p_ac * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[48], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[49], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[50], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[51], p_ab * p_ba * p_ac_inv * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[52], p_ab * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[53], p_ab * p_ba * p_ac_inv * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[54], p_ab * p_ba * p_ac_inv * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[55], p_ab * p_ba * p_ac_inv * p_ca * p_bc * p_cb);
                atomicAdd(&p_motifs[56], p_ab * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[57], p_ab * p_ba * p_ac * p_ca_inv * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[58], p_ab * p_ba * p_ac * p_ca_inv * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[59], p_ab * p_ba * p_ac * p_ca_inv * p_bc * p_cb);
                atomicAdd(&p_motifs[60], p_ab * p_ba * p_ac * p_ca * p_bc_inv * p_cb_inv);
                atomicAdd(&p_motifs[61], p_ab * p_ba * p_ac * p_ca * p_bc_inv * p_cb);
                atomicAdd(&p_motifs[62], p_ab * p_ba * p_ac * p_ca * p_bc * p_cb_inv);
                atomicAdd(&p_motifs[63], p_ab * p_ba * p_ac * p_ca * p_bc * p_cb);
            """,
            'calc_motif_probs'
        )


    def print_kernel_code(self):
        masks, _ = getMotifMasks(3)
        for i, mask in enumerate(masks):
            
            def get_expr(k, not_negated):
                indices = ["ab", "ba", "ac", "ca", "bc", "cb"]
                if(not_negated):
                    return f"p_{indices[k]}"
                else:
                    return f"p_{indices[k]}_inv"

            print(f"atomicAdd(&p_motifs[{i}], {get_expr(0,mask[0])} * {get_expr(1,mask[1])} * {get_expr(2,mask[2])} * {get_expr(3,mask[3])} * {get_expr(4,mask[4])} * {get_expr(5,mask[5])});")


    def get_motif_distribution(self, matrix, discrete, is_probability_matrix = False):
        assert not self.flat_probabilties
        n_rows, n_cols = matrix.shape
        assert n_rows == self.num_neurons
        assert n_cols == self.num_neurons
        
        if(is_probability_matrix):
            prob_matrix = matrix
        else:
            prob_matrix = compute_probability_matrix(matrix, discrete)
        prob_matrix_flat = prob_matrix.flatten()

        p = cp.array(prob_matrix_flat, dtype='float64')        
        I = self.I
        p_motifs = cp.zeros(64, dtype='float64')
        self.calc_motif_probs(I[:,0], I[:,1], I[:,2], I[:,3], I[:,4], I[:,5], p, p_motifs)
        p_motifs_np = p_motifs.get() / I.shape[0]

        p_average_all_pairs = np.mean(prob_matrix)
    
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
            motif_probabilities_64_model[motifKey] = p_motifs_np[i]
            
        motif_probabilities_16_random = aggregateProbabilties_16(motif_probabilities_64_random)
        motif_probabilities_16_model = aggregateProbabilties_16(motif_probabilities_64_model)
        
        deviations = getDeviations(motif_probabilities_16_random, motif_probabilities_16_model, idx_offset=1)
        return deviations
    

    def get_motif_distribution_flat(self, probabilites):
        assert self.flat_probabilties
        assert probabilites.ndim == 2
        assert probabilites.shape[1] == 6
        
        P = cp.array(probabilites, dtype='float64')        
        p_motifs = cp.zeros(64, dtype='float64')
        self.calc_motif_probs_flat(P[:,0], P[:,1], P[:,2], P[:,3], P[:,4], P[:,5], p_motifs)
        p_motifs_np = p_motifs.get() / P.shape[0]

        p_avg = np.mean(probabilites, axis=0)
        p_avg = p_avg.reshape((-1, 6))
        p_avg_inv = 1 - p_avg
            
        motif_probabilities_64_random = {}
        motif_probabilities_64_model = {}
        for i in range(0, self.masks.shape[0]):
            mask = self.masks[i, :]
            mask_inv = self.masks_inv[i, :]
            motifKey = tuple(mask.astype(int))
            motif_probabilities_64_random[motifKey] = calcMotifProbability(p_avg, p_avg_inv, mask, mask_inv)
            motif_probabilities_64_model[motifKey] = p_motifs_np[i]
            
        motif_probabilities_16_random = aggregateProbabilties_16(motif_probabilities_64_random)
        motif_probabilities_16_model = aggregateProbabilties_16(motif_probabilities_64_model)
        
        deviations = getDeviations(motif_probabilities_16_random, motif_probabilities_16_model, idx_offset=1)
        return deviations