import pandas as pd
import numpy as np
import os

def generate_data(n_samples=51):
    """
    Generates synthetic data based on Table 3 statistics from the paper.

    Variables:
    Q: Traffic Volume (Mean=3285.99, Std=2109.43)
    V: Speed (Mean=59.19, Std=17.39)
    PH: Percent Heavy Vehicles (Mean=1.88, Std=1.01)
    G: Gradient (Mean=2.92, Std=2.02)
    D: Density (Mean=0.62, Std=0.26)
    BRF: Building Reflection Factor (Mean=5.41, Std=4.18)
    """
    np.random.seed(42)

    data = {
        'Q': np.abs(np.random.normal(3285.99, 2109.43, n_samples)),
        'V': np.random.normal(59.19, 17.39, n_samples),
        'PH': np.abs(np.random.normal(1.88, 1.01, n_samples)),
        'G': np.abs(np.random.normal(2.92, 2.02, n_samples)),
        'D': np.clip(np.random.normal(0.62, 0.26, n_samples), 0, 1),
        'BRF': np.abs(np.random.normal(5.41, 4.18, n_samples))
    }

    df = pd.DataFrame(data)

    # Generate Target LAeq using Eq (10) from the paper + random noise
    # This creates a realistic relationship between inputs and outputs
    noise = np.random.normal(0, 1.5, n_samples)

    df['LAeq'] = (
        59.826 + 
        (0.001 * df['Q']) + 
        (0.113 * df['V']) + 
        (-0.298 * df['PH']) + 
        (0.057 * df['G']) + 
        (2.115 * df['D']) + 
        (0.170 * df['BRF']) + 
        noise
    )

    # Clip to observed min/max in Table 3 of the paper
    df['LAeq'] = df['LAeq'].clip(lower=64.59, upper=78.52)

    # Ensure directory exists
    os.makedirs('data', exist_ok=True)

    print(f"Synthetic data generated with shape: {df.shape}")
    df.to_csv('data/raw_data.csv', index=False)

if __name__ == "__main__":
    generate_data()
