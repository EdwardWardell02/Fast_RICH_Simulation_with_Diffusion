import numpy as np
import pandas as pd
import yaml
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from physics.RICH_simulator import RICHSimulator
from physics import constants

def _one_hot_encode_particle_type(particle_type):
    """
    Convert particle type string to one-hot encoded vector.
    
    Args:
        particle_type (str): Particle type string
        
    Returns:
        np.array: One-hot encoded vector [pi, K, electron, muon, proton]
    """
    particle_map = {
        'pi': 0,
        'K': 1, 
        'electron': 2,
        'muon': 3,
        'proton': 4
    }
    
    encoding = np.zeros(5)
    if particle_type in particle_map:
        encoding[particle_map[particle_type]] = 1
        
    return encoding

def generate_dataset(config, output_path, num_samples_per_type=10000):
    """
    Generate a dataset of RICH detector events.
    
    Args:
        config (dict): Configuration dictionary
        output_path (str): Path to save the dataset
        num_samples_per_type (int): Number of samples to generate per particle type
    """
    # Initialize simulator
    simulator = RICHSimulator(config)
    
    # Particle types to generate
    particle_types = ['pi', 'K', 'electron', 'muon', 'proton']
    
    # Lists to store data
    all_images = []
    all_types = []
    all_momenta = []
    all_photon_hits = []
    all_one_hot_types = []
    
    # Generate events for each particle type
    for particle_type in particle_types:
        print(f"Generating {num_samples_per_type} events for {particle_type}")
        
        # Generate events
        data = simulator.generate_events(
            num_samples_per_type, 
            particle_type=particle_type
        )
        for i in range(len(data['types'])):
            if data['photon_hits'][i] > 0:
                # Store data
                all_images.append(data['images'][i])
                all_types.append(data['types'][i])
                all_momenta.append(data['momenta'][i])
                all_photon_hits.append(data['photon_hits'][i])
                
                # Create one-hot encoded type
                one_hot_type = _one_hot_encode_particle_type(data['types'][i])
                all_one_hot_types.append(one_hot_type)
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_types = np.array(all_types)
    all_momenta = np.array(all_momenta)
    all_photon_hits = np.array(all_photon_hits)
    all_one_hot_types = np.array(all_one_hot_types)
    
    # Create a DataFrame for metadata
    metadata_df = pd.DataFrame({
        'particle_type': all_types,
        'momentum': all_momenta,
        'photon_hits': all_photon_hits
    })
    
    # Add one-hot encoded columns
    for i, particle in enumerate(['pi', 'K', 'electron', 'muon', 'proton']):
        metadata_df[f'is_{particle}'] = all_one_hot_types[:, i]
    
    # Save data
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save images as numpy array
    np.savez_compressed(
        output_path / 'rich_images.npz',
        images=all_images
    )
    
    # Save metadata as CSV
    metadata_df.to_csv(output_path / 'rich_metadata.csv', index=False)
    
    # Save configuration for reference
    with open(output_path / 'generation_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Dataset generated with {len(all_images)} events")
    print(f"Images saved to: {output_path / 'rich_images.npz'}")
    print(f"Metadata saved to: {output_path / 'rich_metadata.csv'}")