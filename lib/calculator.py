import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

class Calculator:
    def __init__(self, cmin=100, cmax=4):
        self.cmin = cmin
        self.cmax = cmax

    def applyBeta(self, lambdaZero, beta):        
        if(beta < 0):
            coefficient = (1 + beta*(1-1/self.cmin))
        else:
            coefficient = (1 + beta*(self.cmax-1))
        return coefficient * lambdaZero


    def applyActivityCoefficient(self, lambdaZero, beta):  
        meanParam = beta[0]
        stdParam = 4 * np.abs(beta[1])
        
        correlation = np.random.normal(meanParam * np.ones_like(lambdaZero), scale=stdParam)
        lambdaActivity = lambdaZero + correlation
        lambdaActivity[lambdaActivity < 0] = 0
        
        return lambdaActivity


    def applySingleCellSpecificity(self, lambda0, additionalParameters):
        mixingParam1 = additionalParameters[0]
        mixingParam2 = additionalParameters[1]
        randomParameter = additionalParameters[2]
        shapeValues = lambda0.shape
        mixingFactors = softmax([mixingParam1, mixingParam2])

        amplitude = 2
        randomArg = (randomParameter+1)/2 * amplitude
        #lambdaRandom = np.random.poisson(randomArg * np.ones(shapeValues))
        lambdaRandom = np.abs(np.random.normal(0,randomArg,shapeValues))
        lambdaCombined = mixingFactors[0] * lambda0 + mixingFactors[1] * lambdaRandom

        return lambdaCombined

    
    def computeDefaultSpecificity(self, lambdaZero, synapsesEmpirical):        
        if(lambdaZero.size != synapsesEmpirical.size):
            raise ValueError()

        specificity = np.zeros_like(lambdaZero)
        for i in range(0, lambdaZero.size):
            lamb = lambdaZero[i]
            nsyn = synapsesEmpirical[i]
            if(lamb == 0):
                if(nsyn == 0):
                    specificity[i] = 0
                else:
                    specificity[i] = 1
            elif(nsyn < lamb / self.cmin):
                specificity[i] = -1
            elif(nsyn < lamb):
                specificity[i] = (nsyn - lamb) /  (lamb * (1-1/self.cmin))
            elif(nsyn < lamb * self.cmax):
                specificity[i] = (nsyn - lamb) /  (lamb * (self.cmax-1))
            else:
                specificity[i] = 1
        return specificity


if __name__ == "__main__":
    lambdaZero = 10 * np.ones(100)
    calc = Calculator(cmin=1000, cmax=3)

    betas = []
    values = []
    for i in range(-100, 101):
        beta = i / 100
        betas.append(beta)
        values.append(calc.applyBeta(lambdaZero, beta)[0])

    plt.plot(betas, values)
    plt.hlines(10, -1,1, color="black", linestyle="dashed", linewidth=1)
    plt.hlines(10 / calc.cmin, -1,1, color="black", linestyle="dashed", linewidth=1)
    plt.hlines(10 * calc.cmax, -1,1, color="black", linestyle="dashed", linewidth=1)
    plt.text(-0.98,10.3, r"$\lambda_0=PRE_a\cdot POST_b\,/\, POST_{all}$")
    plt.text(-0.98,10 / calc.cmin + 0.3, r"$\lambda_0/c_{min}$")
    plt.text(-0.98,10 * calc.cmax + 0.3, r"$\lambda_0\cdot c_{max}$")
    plt.xlim([-1,1])
    plt.yticks([])
    plt.xlabel(r"specificity $\beta$")
    plt.ylabel(r"expected synapses $ \lambda$")
    plt.savefig("/tmp/scaling.png", dpi=300)

    lamb = 7 * np.ones(5)
    nsyn = np.array([0,35,3.5,9,12])

    print(calc.computeDefaultSpecificity(lamb, nsyn))
