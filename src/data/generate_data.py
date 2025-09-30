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

def generate_dataset(config, output_path, num_samples_per_type=10000, particle_types=None):
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
    if particle_types == None:
        particle_types = ['pi', 'K', 'electron', 'muon', 'proton']
    else:
        particle_types = particle_types
    # Lists to store data
    all_images = []
    all_types = []
    all_momenta = []
    all_photon_hits = []
    all_one_hot_types = []
    
    # Generate events for each particle type
    if particle_types is not None:
        print(f"Generating {num_samples_per_type} events for {particle_types}")
        data = simulator.generate_events(
            num_samples_per_type, 
            particle_type=particle_types
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
    else:           
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

from tqdm import tqdm
"""def generate_combined_events_dataset(config, output_path, number_of_images=1000, num_samples_per_image=10000, particle_type='pi', momentum_distribution='uniform'):
    simulator = RICHSimulator(config)

    all_images = []
    all_photon_hits = []

    for i in tqdm(range(number_of_images)):
        data = simulator.generate_events(
            num_samples_per_image,
            momentum_distribution = momentum_distribution,
            particle_type = particle_type
        )

        # Sum over all generated events to form one combined image
        combined_image = np.sum(data['images'], axis=0)

        # Store
        all_images.append(combined_image)
        all_photon_hits.append(np.sum(data['photon_hits']))

    # Convert to arrays
    all_images = np.array(all_images)
    all_photon_hits = np.array(all_photon_hits)

    # Save results
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path / 'rich_images_combined_events.npz',
        images=all_images,
        photon_hits=all_photon_hits
    )

    with open(output_path / 'generation_config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"Dataset generated with {len(all_images)} combined images")
    print(f"Images saved to: {output_path / 'rich_images_combined_events.npz'}")"""

def generate_combined_events_dataset(config, output_path, number_of_images=1000, num_samples_per_image=1000, particle_type='pi', fixed_momentum=10.0):
    """
    Generate dataset with fixed momentum for consistent mass reconstruction
    """
    simulator = RICHSimulator(config)
    
    all_images = []
    all_photon_hits = []
    
    for i in tqdm(range(number_of_images)):
        # Generate events with FIXED momentum
        images = []
        for _ in range(num_samples_per_image):
            image, hits = simulator.generate_event(particle_type, fixed_momentum)
            images.append(image)
        
        # Sum to create combined image (like your training data)
        combined_image = np.sum(images, axis=0)
        all_images.append(combined_image)
        all_photon_hits.append(np.sum([len(np.where(img > 0)[0]) for img in images]))
    
    # Save dataset
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / f'rich_images_fixed_momentum_{particle_type}_{fixed_momentum}GeV.npz',
        images=np.array(all_images),
        photon_hits=np.array(all_photon_hits)
    )
    
    print(f"Generated {len(all_images)} images with fixed momentum {fixed_momentum} GeV/c for {particle_type}")