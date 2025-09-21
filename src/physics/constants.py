# Particle masses in GeV/c^2
PARTICLE_MASSES = {
    'electron': 0.000511,
    'pi': 0.13957,
    'K': 0.49367,
    'proton': 0.93827,
    'muon': 0.105658
}
# Speed of light in vacuum in m/s
SPEED_OF_LIGHT = 299792458.0  # m/s

# Reduced Planck's constant in GeV·s
H_BAR = 6.582119569e-25

# Fine-structure constant
ALPHA = 1/137.035999084

# Conversion factor for number of Cherenkov photons
# Formula: d²N/dxdλ = 2πα(1 - 1/(β²n²)) / λ²
# Integrated over visible spdectrum (400-700 nm) gives ~370 photons/cm
CHERENKOV_CONSTANT = 370.0  # photons per cm per unit of sin²(θ_c)

