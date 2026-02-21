"""
Probability distribution functions.
Ported from useNetworkStore.ts: checkProbability() and generateRealisticEdgeWeight()
"""
import math
import random


def check_probability(mode: str, p: float) -> bool:
    """
    Returns True with probability p, using the given distribution model.
    Mirrors the TypeScript checkProbability() in useNetworkStore.ts.
    """
    if p <= 0:
        return False
    if p >= 1:
        return True

    if mode == 'Uniform':
        return random.random() < p

    elif mode == 'Normal':
        # Box-Muller transform mapped to [0,1]
        u = 1 - random.random()
        v = random.random()
        z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
        roll = abs(z * 0.15 + 0.5)
        return roll < p

    elif mode == 'Binomial':
        # Simulate 10 trials. p_trial = 1 - (1-p)^(1/10)
        n = 10
        p_trial = 1 - (1 - p) ** (1 / n)
        successes = sum(1 for _ in range(n) if random.random() < p_trial)
        return successes > 0

    elif mode == 'Poisson':
        # P(X >= 1) = 1 - e^-lambda = p  =>  lambda = -ln(1-p)
        lam = -math.log(1 - p)
        L = math.exp(-lam)
        k = 0
        p_val = 1.0
        while True:
            k += 1
            p_val *= random.random()
            if p_val <= L:
                break
        return (k - 1) > 0

    # Fallback
    return random.random() < p


def generate_realistic_edge_weight() -> int:
    """
    Generates a link cost using an Exponential distribution.
    Mirrors generateRealisticEdgeWeight() in useNetworkStore.ts.
    """
    lam = 0.15  # Mean ≈ 1 / 0.15 = 6.67
    u = random.random()
    weight = round(-math.log(1 - u) / lam)
    return max(1, min(30, weight))
