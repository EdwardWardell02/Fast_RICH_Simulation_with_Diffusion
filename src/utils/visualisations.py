import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import sys

sys.path.append('../src')
from physics.RICH_simulator import RICHSimulator
from physics import constants  # Import constants

def _to_numeric_array(image):
    """Coerce image-like input to a numeric numpy array or raise a helpful error."""
    if image is None:
        raise ValueError("Image is None (likely below Cherenkov threshold).")
    # If it's already an ndarray, try to cast dtype
    if isinstance(image, np.ndarray):
        if image.dtype == object:
            # try converting object array -> numeric
            try:
                return image.astype(float)
            except Exception:
                # fallback: convert via list then to float
                return np.array(image.tolist(), dtype=float)
        else:
            return image.astype(float)
    # If it's a list of lists or similar
    try:
        return np.array(image, dtype=float)
    except Exception as exc:
        raise TypeError("Could not coerce image to numeric numpy array.") from exc
    
def plot_rich_image(image, title="RICH Detector Image", cmap='viridis'):
    """
    Plot a single RICK detector image.
    
    Args:
        image (np.ndarray): 2D array representing the detector image
        title (str): Plot title
        cmap (str): Colourmap
    """
    img = _to_numeric_array(image)
    plt.figure(figsize=(8,6))
    plt.imshow(image, cmap=cmap, origin='lower')
    plt.colorbar(label='Photon Count')
    plt.title(title)
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

def plot_rich_images(images, title="RICH Detector Image", cmap='viridis'):

    combined = np.zeros_like(images[0], dtype=float)

    # Sum all images (overlay photon hits)
    for img in images:
        combined += img

    plt.figure(figsize=(8, 6))
    plt.imshow(combined, cmap=cmap, origin='lower')
    plt.colorbar(label='Photon Count')
    plt.title(title)
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

    
def plot_cherenkov_angle_vs_momentum(simulator, num_events=1000, momentum_range=[0, 100], momentum_distribution='1/p2'):
    """
    Plot measured Cherenkov angle vs true momentum for individual events,
    color-coded by particle type.
    
    Args:
        simulator (RICHSimulator): An instance of the RICH simulator
        num_events (int): Number of events to generate
        momentum_range (list): [min, max] momentum range in GeV/c
    """
    # Define particle types and colors
    particle_types = ['pi', 'K', 'proton', 'electron', 'muon']
    colours = {'pi': 'red', 'K': 'blue', 'proton': 'green', 'electron': 'purple', 'muon': 'teal'}
    
    # Generate events with random parameters
    events = simulator.generate_events(num_events, momentum_range, None, momentum_distribution)
        
    plt.figure(figsize=(10, 6))
    
    # Plot each event individually
    for i in range(len(events['types'])):
        particle = events['types'][i]
        momentum = events['momenta'][i]
        image = events['images'][i]
        photon_hits = events['photon_hits'][i]
        
        # Calculate the measured Cherenkov angle from the image
        # This simulates how LHCb would reconstruct the angle from the ring
        measured_angle = simulator.reconstruct_angle_from_image(image)
        
        if measured_angle is not None:  # Only plot if we successfully reconstructed an angle
            plt.scatter(momentum, measured_angle, 
                        color=colours[particle], 
                        alpha=0.6,
                        s=10)  # Smaller points to see density
    
    # Create legend manually since we're plotting points individually
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colours['pi'], markersize=8, label='π'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colours['K'], markersize=8, label='K'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colours['proton'], markersize=8, label='p'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colours['electron'], markersize=8, label='e'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=colours['muon'], markersize=8, label='μ')]
    
    # Format the plot
    plt.xlabel('True Momentum (GeV/c)', fontsize=16)
    plt.ylabel('Measured Cherenkov Angle (radians)', fontsize=16)
    plt.title(f'Cherenkov Angle vs Momentum for n = {simulator.n}', fontsize=20)
    plt.legend(handles=legend_elements, loc='lower right', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    ax = plt.gca()
    ax.set_xticks([0,25,50,75,100])
    ax.set_yticks([0, 0.01, 0.02, 0.03])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.002))
    ax.tick_params(axis='both', which='minor', length=3, width=0.8, direction='in', labelsize=17)
    ax.tick_params(axis='both', which='major', length=6, width=1.2, direction='in', labelsize=12)
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', visible=False)
    # Set appropriate axis limits
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()

# Check the photon count, plot mean number of detected photons per event vs momentum.
# Should follow sin^2(theta_c)

def photon_count(simulator, num_events=1000, particle_types=None, momentum_range=[0.01, 100]):
    """
    Plot photon hits vs momentum for specified particle types.
    
    Args:
        simulator (RICHSimulator): An instance of the RICH simulator
        num_events (int): Number of events to generate per particle type
        particle_types (list or str or None): Particle types to include. 
            If None, include all available types.
        momentum_range (list): [min, max] momentum range in GeV/c
    """
    # Define particle colours
    colours = {'pi': 'red', 'K': 'blue', 'p': 'green', 'electron': 'purple', 'muon': 'teal'}
    
    # Handle different input formats for particle_types
    if particle_types is None:
        # Use all available particle types
        particle_types = list(constants.PARTICLE_MASSES.keys())
    elif isinstance(particle_types, str):
        # Single particle type as string
        particle_types = [particle_types]
    
    # Validate particle types
    for p_type in particle_types:
        if p_type not in constants.PARTICLE_MASSES:
            raise ValueError(f"Invalid particle type: {p_type}")
    
    plt.figure(figsize=(10, 6))
    
    # Generate and plot events for each particle type
    legend_elements = []
    
    for p_type in particle_types:
        # Generate events for this particle type
        events = simulator.generate_events(num_events, momentum_range, particle_type=p_type)
        
        # Plot each event
        for i in range(len(events['types'])):
            momentum = events['momenta'][i]
            photon_hits = events['photon_hits'][i]
            
            if photon_hits is not None:
                plt.scatter(momentum, photon_hits, 
                            color=colours[p_type], 
                            alpha=0.6,
                            s=10)
        
        # Add to legend
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colours[p_type], 
                                     markersize=8, label=p_type))
    
    # Format the plot
    plt.xlabel('True Momentum (GeV/c)')
    plt.ylabel('Number of Photon Hits')
    plt.title(f'Photon Hits per event vs Momentum for n = {simulator.n}')
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    ax = plt.gca()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='both', which='minor', length=3, width=0.8, direction='in')
    ax.tick_params(axis='both', which='major', length=6, width=1.2, direction='in')
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', visible=False)
    
    # Set appropriate axis limits
    plt.xlim(momentum_range[0], momentum_range[1])
    plt.tight_layout()
    plt.show()



def display_image_grid(images, num_images=9, figsize=(12, 12)):
    """
    Display a grid of RICH images.
    
    Parameters:
    images (numpy array): Array of images
    num_images (int): Number of images to display (default: 9)
    figsize (tuple): Figure size (default: (12, 12))
    """
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    # Flatten axes array for easy iteration
    axes = axes.ravel()
    
    # Display images
    for i in range(num_images):
        if i < len(images):
            axes[i].imshow(images[i], cmap='viridis')
            axes[i].set_title(f'Image {i+1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')  # Hide empty subplots
    
    # Hide any remaining empty subplots
    for i in range(num_images, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()