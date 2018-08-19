from numba import guvectorize
from numpy import shape, array
from numpy.linalg import det


def VKE_TETRA_E(E, nu, volume, beta1, beta2, beta3, beta4,
              gama1, gama2, gama3, gama4,
              delta1, delta2, delta3, delta4):

    from numpy import array, zeros

    vke = zeros(24**2, float)

    IndexVkeInVkeE = array([ 
            0,   1,   2,   6,   7,   8,  12,  13,  14,  18,  19,  20,  24,
            25,  26,  30,  31,  32,  36,  37,  38,  42,  43,  44,  48,  49,
            50,  54,  55,  56,  60,  61,  62,  66,  67,  68, 144, 145, 146,
            150, 151, 152, 156, 157, 158, 162, 163, 164, 168, 169, 170, 174,
            175, 176, 180, 181, 182, 186, 187, 188, 192, 193, 194, 198, 199,
            200, 204, 205, 206, 210, 211, 212, 288, 289, 290, 294, 295, 296,
            300, 301, 302, 306, 307, 308, 312, 313, 314, 318, 319, 320, 324,
            325, 326, 330, 331, 332, 336, 337, 338, 342, 343, 344, 348, 349,
            350, 354, 355, 356, 432, 433, 434, 438, 439, 440, 444, 445, 446,
            450, 451, 452, 456, 457, 458, 462, 463, 464, 468, 469, 470, 474,
            475, 476, 480, 481, 482, 486, 487, 488, 492, 493, 494, 498, 499,
            500])

    vke[IndexVkeInVkeE] = array([[0.0277777777777778 * E * (beta1**2 * (nu - 1.0) + delta1**2 * (1.0 * nu - 0.5) + gama1**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta1 * gama1 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta1 * delta1 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta2 * (nu - 1.0) + delta1 * delta2 * (1.0 * nu - 0.5) + gama1 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * gama2 * nu + beta2 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * delta2 * nu + beta2 * delta1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta3 * (nu - 1.0) + delta1 * delta3 * (1.0 * nu - 0.5) + gama1 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * gama3 * nu + beta3 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * delta3 * nu + beta3 * delta1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta4 * (nu - 1.0) + delta1 * delta4 * (1.0 * nu - 0.5) + gama1 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * gama4 * nu + beta4 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * delta4 * nu + beta4 * delta1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta1 * gama1 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1**2 * (1.0 * nu - 0.5) + delta1**2 * (1.0 * nu - 0.5) + gama1**2 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta1 * gama1 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * gama2 * (1.0 * nu - 0.5) - beta2 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta2 * (1.0 * nu - 0.5) + delta1 * delta2 * (1.0 * nu - 0.5) + gama1 * gama2 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta1 * gama2 * (1.0 * nu - 0.5) - delta2 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * gama3 * (1.0 * nu - 0.5) - beta3 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta3 * (1.0 * nu - 0.5) + delta1 * delta3 * (1.0 * nu - 0.5) + gama1 * gama3 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta1 * gama3 * (1.0 * nu - 0.5) - delta3 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * gama4 * (1.0 * nu - 0.5) - beta4 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta4 * (1.0 * nu - 0.5) + delta1 * delta4 * (1.0 * nu - 0.5) + gama1 * gama4 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta1 * gama4 * (1.0 * nu - 0.5) - delta4 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta1 * delta1 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta1 * gama1 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1**2 * (1.0 * nu - 0.5) + delta1**2 * (nu - 1.0) + gama1**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * delta2 * (1.0 * nu - 0.5) - beta2 * delta1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta1 * gama2 * nu + delta2 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta2 * (1.0 * nu - 0.5) + delta1 * delta2 * (nu - 1.0) + gama1 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * delta3 * (1.0 * nu - 0.5) - beta3 * delta1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta1 * gama3 * nu + delta3 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta3 * (1.0 * nu - 0.5) + delta1 * delta3 * (nu - 1.0) + gama1 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * delta4 * (1.0 * nu - 0.5) - beta4 * delta1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta1 * gama4 * nu + delta4 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta4 * (1.0 * nu - 0.5) + delta1 * delta4 * (nu - 1.0) + gama1 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta2 * (nu - 1.0) + delta1 * delta2 * (1.0 * nu - 0.5) + gama1 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * gama2 * (1.0 * nu - 0.5) - beta2 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * delta2 * (1.0 * nu - 0.5) - beta2 * delta1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2**2 * (nu - 1.0) + delta2**2 * (1.0 * nu - 0.5) + gama2**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta2 * gama2 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta2 * delta2 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta3 * (nu - 1.0) + delta2 * delta3 * (1.0 * nu - 0.5) + gama2 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * gama3 * nu + beta3 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * delta3 * nu + beta3 * delta2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta4 * (nu - 1.0) + delta2 * delta4 * (1.0 * nu - 0.5) + gama2 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * gama4 * nu + beta4 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * delta4 * nu + beta4 * delta2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * gama2 * nu + beta2 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta2 * (1.0 * nu - 0.5) + delta1 * delta2 * (1.0 * nu - 0.5) + gama1 * gama2 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta1 * gama2 * nu + delta2 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta2 * gama2 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2**2 * (1.0 * nu - 0.5) + delta2**2 * (1.0 * nu - 0.5) + gama2**2 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta2 * gama2 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * gama3 * (1.0 * nu - 0.5) - beta3 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta3 * (1.0 * nu - 0.5) + delta2 * delta3 * (1.0 * nu - 0.5) + gama2 * gama3 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta2 * gama3 * (1.0 * nu - 0.5) - delta3 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * gama4 * (1.0 * nu - 0.5) - beta4 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta4 * (1.0 * nu - 0.5) + delta2 * delta4 * (1.0 * nu - 0.5) + gama2 * gama4 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta2 * gama4 * (1.0 * nu - 0.5) - delta4 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * delta2 * nu + beta2 * delta1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta1 * gama2 * (1.0 * nu - 0.5) - delta2 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta2 * (1.0 * nu - 0.5) + delta1 * delta2 * (nu - 1.0) + gama1 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta2 * delta2 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta2 * gama2 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2**2 * (1.0 * nu - 0.5) + delta2**2 * (nu - 1.0) + gama2**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * delta3 * (1.0 * nu - 0.5) - beta3 * delta2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta2 * gama3 * nu + delta3 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta3 * (1.0 * nu - 0.5) + delta2 * delta3 * (nu - 1.0) + gama2 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * delta4 * (1.0 * nu - 0.5) - beta4 * delta2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta2 * gama4 * nu + delta4 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta4 * (1.0 * nu - 0.5) + delta2 * delta4 * (nu - 1.0) + gama2 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta3 * (nu - 1.0) + delta1 * delta3 * (1.0 * nu - 0.5) + gama1 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * gama3 * (1.0 * nu - 0.5) - beta3 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * delta3 * (1.0 * nu - 0.5) - beta3 * delta1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta3 * (nu - 1.0) + delta2 * delta3 * (1.0 * nu - 0.5) + gama2 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * gama3 * (1.0 * nu - 0.5) - beta3 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * delta3 * (1.0 * nu - 0.5) - beta3 * delta2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3**2 * (nu - 1.0) + delta3**2 * (1.0 * nu - 0.5) + gama3**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta3 * gama3 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta3 * delta3 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * beta4 * (nu - 1.0) + delta3 * delta4 * (1.0 * nu - 0.5) + gama3 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta3 * gama4 * nu + beta4 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta3 * delta4 * nu + beta4 * delta3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * gama3 * nu + beta3 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta3 * (1.0 * nu - 0.5) + delta1 * delta3 * (1.0 * nu - 0.5) + gama1 * gama3 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta1 * gama3 * nu + delta3 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * gama3 * nu + beta3 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta3 * (1.0 * nu - 0.5) + delta2 * delta3 * (1.0 * nu - 0.5) + gama2 * gama3 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta2 * gama3 * nu + delta3 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta3 * gama3 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3**2 * (1.0 * nu - 0.5) + delta3**2 * (1.0 * nu - 0.5) + gama3**2 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta3 * gama3 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * gama4 * (1.0 * nu - 0.5) - beta4 * gama3 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * beta4 * (1.0 * nu - 0.5) + delta3 * delta4 * (1.0 * nu - 0.5) + gama3 * gama4 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta3 * gama4 * (1.0 * nu - 0.5) - delta4 * gama3 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * delta3 * nu + beta3 * delta1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta1 * gama3 * (1.0 * nu - 0.5) - delta3 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta3 * (1.0 * nu - 0.5) + delta1 * delta3 * (nu - 1.0) + gama1 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * delta3 * nu + beta3 * delta2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta2 * gama3 * (1.0 * nu - 0.5) - delta3 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta3 * (1.0 * nu - 0.5) + delta2 * delta3 * (nu - 1.0) + gama2 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta3 * delta3 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta3 * gama3 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3**2 * (1.0 * nu - 0.5) + delta3**2 * (nu - 1.0) + gama3**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * delta4 * (1.0 * nu - 0.5) - beta4 * delta3 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta3 * gama4 * nu + delta4 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * beta4 * (1.0 * nu - 0.5) + delta3 * delta4 * (nu - 1.0) + gama3 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta4 * (nu - 1.0) + delta1 * delta4 * (1.0 * nu - 0.5) + gama1 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * gama4 * (1.0 * nu - 0.5) - beta4 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * delta4 * (1.0 * nu - 0.5) - beta4 * delta1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta4 * (nu - 1.0) + delta2 * delta4 * (1.0 * nu - 0.5) + gama2 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * gama4 * (1.0 * nu - 0.5) - beta4 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * delta4 * (1.0 * nu - 0.5) - beta4 * delta2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * beta4 * (nu - 1.0) + delta3 * delta4 * (1.0 * nu - 0.5) + gama3 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * gama4 * (1.0 * nu - 0.5) - beta4 * gama3 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * delta4 * (1.0 * nu - 0.5) - beta4 * delta3 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta4**2 * (nu - 1.0) + delta4**2 * (1.0 * nu - 0.5) + gama4**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta4 * gama4 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta4 * delta4 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * gama4 * nu + beta4 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta4 * (1.0 * nu - 0.5) + delta1 * delta4 * (1.0 * nu - 0.5) + gama1 * gama4 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta1 * gama4 * nu + delta4 * gama1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * gama4 * nu + beta4 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta4 * (1.0 * nu - 0.5) + delta2 * delta4 * (1.0 * nu - 0.5) + gama2 * gama4 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta2 * gama4 * nu + delta4 * gama2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta3 * gama4 * nu + beta4 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * beta4 * (1.0 * nu - 0.5) + delta3 * delta4 * (1.0 * nu - 0.5) + gama3 * gama4 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-delta3 * gama4 * nu + delta4 * gama3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta4 * gama4 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta4**2 * (1.0 * nu - 0.5) + delta4**2 * (1.0 * nu - 0.5) + gama4**2 * (nu - 1.0)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta4 * gama4 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta1 * delta4 * nu + beta4 * delta1 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta1 * gama4 * (1.0 * nu - 0.5) - delta4 * gama1 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta1 * beta4 * (1.0 * nu - 0.5) + delta1 * delta4 * (nu - 1.0) + gama1 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta2 * delta4 * nu + beta4 * delta2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta2 * gama4 * (1.0 * nu - 0.5) - delta4 * gama2 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta2 * beta4 * (1.0 * nu - 0.5) + delta2 * delta4 * (nu - 1.0) + gama2 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (-beta3 * delta4 * nu + beta4 * delta3 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (delta3 * gama4 * (1.0 * nu - 0.5) - delta4 * gama3 * nu) / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta3 * beta4 * (1.0 * nu - 0.5) + delta3 * delta4 * (nu - 1.0) + gama3 * gama4 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * beta4 * delta4 / (volume * (nu + 1) * (2 * nu - 1)),
                  -0.0138888888888889 * E * delta4 * gama4 / (volume * (nu + 1) * (2 * nu - 1)),
                  0.0277777777777778 * E * (beta4**2 * (1.0 * nu - 0.5) + delta4**2 * (nu - 1.0) + gama4**2 * (1.0 * nu - 0.5)) / (volume * (nu + 1) * (2 * nu - 1))]],
                float)

    vke[[75, 100, 125, 225, 250, 275, 375, 400, 425, 525, 550, 575]] = 1.e-10 

    return vke


def VKG_TETRA_E(E, nu, coord, connec_volume, arrayfalse, out):
    
    nglel = 24
    
    p, q = shape(coord)
    pp, qq = shape(connec_volume)
    k = len(arrayfalse)

    nelem = pp

    tt = int(nglel**2)
    for i in range(nelem):
        x0, x1, x2, x3 = coord[connec_volume[i], 0]
        y0, y1, y2, y3 = coord[connec_volume[i], 1]
        z0, z1, z2, z3 = coord[connec_volume[i], 2]

        volume = abs((1. / 6.) * det(array([[1., x0, y0, z0],
                                            [1., x1, y1, z1],
                                            [1., x2, y2, z2],
                                            [1., x3, y3, z3]])))

        beta1 = -det(array([[1., y1, z1], [1., y2, z2], [1., y3, z3]]))
        beta2 = det(array([[1., y0, z0], [1., y2, z2], [1., y3, z3]]))
        beta3 = -det(array([[1., y0, z0], [1., y1, z1], [1., y3, z3]]))
        beta4 = det(array([[1., y0, z0], [1., y1, z1], [1., y2, z2]]))

        gama1 = det(array([[1., x1, z1], [1., x2, z2], [1., x3, z3]]))
        gama2 = -det(array([[1., x0, z0], [1., x2, z2], [1., x3, z3]]))
        gama3 = det(array([[1., x0, z0], [1., x1, z1], [1., x3, z3]]))
        gama4 = -det(array([[1., x0, z0], [1., x1, z1], [1., x2, z2]]))

        delta1 = -det(array([[1., x1, y1], [1., x2, y2], [1., x3, y3]]))
        delta2 = det(array([[1., x0, y0], [1., x2, y2], [1., x3, y3]]))
        delta3 = -det(array([[1., x0, y0], [1., x1, y1], [1., x3, y3]]))
        delta4 = det(array([[1., x0, y0], [1., x1, y1], [1., x2, y2]]))

        ve = VKE_TETRA_E(E, nu, volume, beta1, beta2, beta3, beta4,
                       gama1, gama2, gama3, gama4,
                       delta1, delta2, delta3, delta4)

        start = i * tt
        end = (i + 1) * tt
        out[start:end] = ve


GU_VKG_TETRA_E = guvectorize(['float32, float32, float32[:,:],\
                          int32[:,:], int32[:], float32[:]',
                                     'float64, float64, float64[:,:],\
                           int64[:,:], int64[:], float64[:]'],
                                    '(),(),(p,q),(pp,qq),(k)->(k)')(VKG_TETRA_E)


def KG_TETRA_E(E, nu, ngl, coord, connec_volume, I, J):
    from numpy import empty
    from scipy.sparse import csr_matrix

    arrayfalse = empty(I.size, int)

    vkg = GU_VKG_TETRA_E(E, nu, coord, connec_volume, arrayfalse)

    kg = csr_matrix((vkg, (I, J)), shape=(ngl, ngl))

    return kg