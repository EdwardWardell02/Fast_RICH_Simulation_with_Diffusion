import numpy as np
from . import constants

class RICHSimulator:
    def __init__(self, config):
        self.n = config['simulator']['refractive_index']
        self.L = config['simulator']['radiator_length']
        self.angular_resolution = config['simulator']['single_photon_resolution'] * 1e-3
        self.epsilon = config['simulator']['detection_efficiency']
        self.image_size = config['simulator']['image_size']
        self.pixel_pitch = config['simulator']['pixel_pitch']
        self.detector_distance = config['simulator']['detector_distance'] #Distance from radiator to detector in meters

        self.detector_center = self.image_size / 2
        self.detector_bounds = [0, self.image_size]

    def _cherenkov_angle(self, p, mass):
        """
        Calculate the Cherenkov angle θ_c given the particle's momentum and mass and
        the refractive index n.

        Args:
            p (float): Particle momentum in GeV/c.
            m (float): Particle mass in GeV/c^2.
        
        Returns:
            float: Cherenkov angle in radians. None if below threshold.
        """

        # FInd the particle's velocity β
        E = np.sqrt(p**2 + mass**2)  # Total energy in GeV
        beta = p / E  # Dimensionless

        if beta * self.n < 1:
            return None  # Below Cherenkov threshold
        
        theta_c = np.arccos(1 / (beta * self.n))

        return theta_c
    
    def _calculate_photon_yield_frank_tamm(self, theta_c, L=None, lam_min=300e-9,lam_max=600e-9):
        """
        Calculate the expected number of Cherenkov photons produced usign Frank-Tamm formula.

        Args:
            theta_c (float): Cherenkov angle in radians.
        
        Returns:
            float: Expected number of photons (Poisson distributed around the mean).
        """
        if L is None:
            L = self.L
        
        # boundary case
        if theta_c is None:
            return 0, 0.0
        
        sin2 = np.sin(theta_c)**2

        #Integration grid
        n_steps = 1000
        lams = np.linspace(lam_min, lam_max, n_steps)
        # Differential yield: 2π α / λ^2 * sin^2 θ
        alpha = constants.ALPHA
        two_pi_alpha = 2.0*np.pi*alpha
        dndlam = two_pi_alpha * sin2 / (lams**2)
        
        # Mean number of photons
        mean_per_m = np.trapz(dndlam, lams)  # photons per meter
        mean_photons = mean_per_m * L
        
        # Poisson fluctuate
        detected = np.random.poisson(mean_photons)
        
        return detected, mean_photons

    
    def _generate_photon_hits(self, theta_c, num_photons):
        """
        Generate photon hit positions on the detector plane.

        Args:
            theta_c (float): Cherenkov angle in radians.
            num_photons (int): Number of photons to simulate.
        
        Returns:
            list of tuples: List of (x, y) positions of photon hits on the detector.
        """
        hits = []

        for _ in range(num_photons):
            # Sample azimuthal angle uniformly
            phi = np.random.uniform(0, 2 * np.pi)

            # Apply angular resolution smearing to Cherenkov angle
            theta_smeared = np.random.normal(theta_c, self.angular_resolution)
            
            # Calculate hit position on detector
            r = self.detector_distance * np.tan(theta_smeared)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            # Apply detection efficiency
            if np.random.random() < self.epsilon:
                hits.append((x, y))
            
        return hits
    
    def _sample_momentum_1_over_p2(self, p_min, p_max, size=1):
        """
        Sample momentum from a 1/p^2 distribution between p_min and p_max.
        
        Args:
            p_min (float): Minimum momentum
            p_max (float): Maximum momentum
            size (int): Number of samples to generate
        
        Returns:
            numpy.ndarray: Array of sampled momenta
        """
        # Normalization constant
        norm = 1/p_min - 1/p_max
        
        # Generate uniform random numbers
        u = np.random.uniform(0, 1, size)
        
        # Apply inverse transform sampling
        p_samples = 1 / (1/p_min - u * norm)
        
        return p_samples
    def _sample_momentum_exponential(self, p_min, p_max, scale=5.0, size=1):
        """
        Sample momentum from an exponential distribution truncated between p_min and p_max.
        
        Args:
            p_min (float): Minimum momentum
            p_max (float): Maximum momentum
            scale (float): Scale parameter for exponential distribution
            size (int): Number of samples to generate
        
        Returns:
            numpy.ndarray: Array of sampled momenta
        """
        # Generate exponential samples
        samples = np.random.exponential(scale, size*10)  # Generate extra to account for truncation
        
        # Truncate to desired range
        samples = samples[(samples >= p_min) & (samples <= p_max)]
        
        # If we don't have enough samples, generate more
        while len(samples) < size:
            additional = np.random.exponential(scale, size*10)
            additional = additional[(additional >= p_min) & (additional <= p_max)]
            samples = np.concatenate([samples, additional])
        
        return samples[:size]
    
    def _digitise_hits(self, hits):
        """
        Convert continuous hit positions to pixel indices.

        Args:
            hits (list of tuples): List of (x, y) positions of photon hits.
        
        Returns:
            list of tuples: List of (i, j) pixel indices.
        """
        image = np.zeros((self.image_size, self.image_size), dtype=int)
        for x,y in hits:
            # Convert to pixel indices (center of detector is as (detector_center, detector_center))
            i = int(self.detector_center + x / self.pixel_pitch)
            j = int(self.detector_center + y / self.pixel_pitch)

            # Check bounds
            if (0 <= i < self.image_size) and (0 <= j < self.image_size):
                image[i, j] += 1

        return image
    
    def generate_event(self, particle_type, momentum):
        """
        Generate a RICH detector event for a given particle type and momentum.

        Args:
            particle_type (str): Type of the particle ('electron', 'pi', 'K', 'p').
            momentum (float): Particle momentum in GeV/c.
        Returns:
            np.ndarray: 2D array representing the digitised detector image, or None if below threshold.
        """

        # get particle mass
        mass = constants.PARTICLE_MASSES[particle_type]

        # calculate Cherenkov angle
        theta_c = self._cherenkov_angle(momentum, mass)

        #print(f"Particle: {particle_type}, Momentum: {momentum} GeV/c")
        #print(f"Cherenkov angle: {np.degrees(theta_c) if theta_c else 'None'} degrees")

        if theta_c is None:
            # Return zero image if below threshold
            image = np.zeros((self.image_size, self.image_size), dtype=int)
            return image, 0
        
        # calculate expected photon yield
        num_photons, mean_photons = self._calculate_photon_yield_frank_tamm(theta_c)
        #print(f"Number of photons: {num_photons} (mean: {mean_photons})")

        # generate photon hits
        hits = self._generate_photon_hits(theta_c, num_photons)
        #print(f"Number of detected hits: {len(hits)}")
        # digitise hits to form image
        image = self._digitise_hits(hits)
        photon_hits = len(hits)
        
        return image, photon_hits
    
    def generate_events(self, num_events, momentum_range=None, particle_type=None, momentum_distribution='uniform'):
        """
        Generate multiple RICH detector events with random parameters.
        
        Args:
            num_events (int): Number of events to generate
            momentum_range (list): [min, max] momentum range in GeV/c
            particle_type (str or list): Specific particle type(s) to generate
            momentum_distribution (str): Distribution to use for momentum sampling
        """
        # Handle particle_type input - FIXED VERSION
        if particle_type is None:
            particle_types = list(constants.PARTICLE_MASSES.keys())
        elif isinstance(particle_type, str):
            particle_types = [particle_type]
        elif isinstance(particle_type, (list, tuple, set, np.ndarray)):
            particle_types = list(particle_type)
            # Validate particle types
            invalid = [p for p in particle_types if p not in constants.PARTICLE_MASSES]
            if invalid:
                raise ValueError(f"Invalid particle type(s): {invalid}")
        else:
            raise TypeError("particle_type must be None, a string, or a list/tuple/set/ndarray of strings")

        images = []
        types = []
        momenta = []
        photon_hits = []
        
        for _ in range(num_events):
            # Select particle type - FIXED: Always choose randomly from available types
            p_type = np.random.choice(particle_types)
                
            # Determine momentum range
            if momentum_range is None:
                if p_type == 'K':
                    p_range = [2.0, 100.0]
                elif p_type == 'pi':
                    p_range = [0.6, 100.0]
                else:
                    p_range = [2.0, 100.0]
            else:
                p_range = momentum_range
                
            # Sample momentum
            if momentum_distribution == 'uniform':
                p = np.random.uniform(p_range[0], p_range[1])
            elif momentum_distribution == '1/p2':
                p = self._sample_momentum_1_over_p2(p_range[0], p_range[1])
            elif momentum_distribution == 'exponential':
                p = self._sample_momentum_exponential(p_range[0], p_range[1], scale=3.0)
            else:
                raise ValueError(f"Unknown momentum distribution: {momentum_distribution}")

            # Generate image
            image, hits = self.generate_event(p_type, p)
            images.append(image)
            types.append(p_type)
            momenta.append(p)
            photon_hits.append(hits)

        return {
            'images': np.array(images),
            'types': np.array(types),
            'momenta': np.array(momenta),
            'photon_hits': np.array(photon_hits)
        }
    
    def reconstruct_angle_from_image(self, image):
        """
        Reconstruct the Cherenkov angle from a detector image.
        This simulates how LHCb would analyze the ring pattern.
        
        Args:
            image (np.ndarray): 2D array representing the detector image
            simulator (RICHSimulator): The simulator instance
        
        Returns:
            float: Reconstructed Cherenkov angle in radians, or None if reconstruction fails
        """
        # Find non-zero pixels (photon hits)
        y_indices, x_indices = np.where(image > 0)
        
        # Skip if too few hits for reliable reconstruction
        if len(x_indices) < 5:
            return None
        
        # Convert pixel indices to physical coordinates
        x_positions = (x_indices - self.detector_center) * self.pixel_pitch
        y_positions = (y_indices - self.detector_center) * self.pixel_pitch
        
        # Calculate distances from center
        distances = np.sqrt(x_positions**2 + y_positions**2)
        
        # Calculate the mean radius (this is a simplified reconstruction)
        mean_radius = np.mean(distances)
        
        # Convert radius to angle: tan(θ) = r / d
        reconstructed_angle = np.arctan(mean_radius / self.detector_distance)
        
        return reconstructed_angle