# System Information and Hardware Specifications
import psutil
import platform
from datetime import datetime

def display_system_information():
    """
    Display comprehensive system information for reproducibility.
    
    Returns:
        dict: System specifications including CPU, memory, and platform details
    """
    cpu_frequency = psutil.cpu_freq().current if psutil.cpu_freq() else "Unknown"
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    system_specs = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_frequency_mhz': cpu_frequency,
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'total_ram_gb': round(total_ram_gb, 2)
    }
    
    print("=== SYSTEM SPECIFICATIONS ===")
    for key, value in system_specs.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=" * 30)
    
    return system_specs

# Display system information
system_info = display_system_information()
