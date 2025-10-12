"""
Synapse Cycle Module for Keithley Sourcemeter
Implements potentiation-depression cycling for reproducibility assessment.
Allows independent control of potentiation and depression parameters.

Enhanced Features:
- Parameter presets for common protocols
- Real-time plot updates during measurement
- Cycle-specific read voltages
- Multi-device sequential testing
"""

import numpy as np
import time
from tkinter import messagebox
import synapse_engine

# =============================================================================
# CORE CYCLING FUNCTIONS
# =============================================================================

def run_potentiation_depression_cycle(instrument, stim_ch, read_ch, pot_params, dep_params, n_cycles, 
                                     inter_cycle_delay_ms=1000, update_callback=None):
    """
    Runs alternating potentiation and depression cycles with real-time updates.
    
    Each cycle consists of:
    1. Potentiation phase (LTP) with pot_params
    2. Depression phase (LTD) with dep_params
    
    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Stimulus channel ('smua' or 'smub')
        read_ch (str): Read channel ('smua' or 'smub')
        pot_params (dict): Potentiation parameters (see synapse_engine.pulse_read_sequence)
        dep_params (dict): Depression parameters (see synapse_engine.pulse_read_sequence)
        n_cycles (int): Number of complete potentiation-depression cycles
        inter_cycle_delay_ms (float): Delay between cycles in milliseconds
        update_callback (callable, optional): Function to call after each cycle with results
    
    Returns:
        dict: Results containing:
            - cycles: List of cycle results, each with 'potentiation' and 'depression' keys
            - summary_metrics: Aggregated metrics across all cycles
            - pot_params: Copy of potentiation parameters
            - dep_params: Copy of depression parameters
            - n_cycles: Number of cycles performed
    """
    results = {
        "cycles": [],
        "pot_params": pot_params.copy(),
        "dep_params": dep_params.copy(),
        "n_cycles": n_cycles,
        "inter_cycle_delay_ms": inter_cycle_delay_ms
    }
    
    for cycle_idx in range(n_cycles):
        print(f"\n{'='*50}")
        print(f"Starting Cycle {cycle_idx + 1}/{n_cycles}")
        print(f"{'='*50}")
        
        cycle_data = {
            "cycle_number": cycle_idx + 1,
            "potentiation": None,
            "depression": None
        }
        
        # === POTENTIATION PHASE ===
        print(f"\n--- Potentiation Phase ---")
        try:
            pot_results = synapse_engine.pulse_read_sequence(
                instrument, stim_ch, read_ch, pot_params
            )
            cycle_data["potentiation"] = pot_results
            
            # Calculate and display potentiation metrics
            pot_metrics = synapse_engine.calculate_synapse_metrics(pot_results)
            print(f"Potentiation ΔG: {pot_metrics.get('Delta G (%)', 'N/A')}%")
            
        except Exception as e:
            print(f"⚠️ Potentiation phase failed: {str(e)}")
            cycle_data["potentiation"] = {"error": str(e)}
        
        # Small delay between phases
        time.sleep(0.5)
        
        # === DEPRESSION PHASE ===
        print(f"\n--- Depression Phase ---")
        try:
            dep_results = synapse_engine.pulse_read_sequence(
                instrument, stim_ch, read_ch, dep_params
            )
            cycle_data["depression"] = dep_results
            
            # Calculate and display depression metrics
            dep_metrics = synapse_engine.calculate_synapse_metrics(dep_results)
            print(f"Depression ΔG: {dep_metrics.get('Delta G (%)', 'N/A')}%")
            
        except Exception as e:
            print(f"⚠️ Depression phase failed: {str(e)}")
            cycle_data["depression"] = {"error": str(e)}
        
        # Store cycle data
        results["cycles"].append(cycle_data)
        
        # Real-time update callback
        if update_callback is not None:
            try:
                update_callback(results)
            except Exception as e:
                print(f"⚠️ Update callback failed: {str(e)}")
        
        # Inter-cycle delay (except after last cycle)
        if cycle_idx < n_cycles - 1:
            print(f"\nWaiting {inter_cycle_delay_ms} ms before next cycle...")
            time.sleep(inter_cycle_delay_ms / 1000.0)
    
    # Calculate summary metrics across all cycles
    results["summary_metrics"] = calculate_cycle_summary_metrics(results)
    
    return results


def simulate_potentiation_depression_cycle(pot_params, dep_params, n_cycles, update_callback=None):
    """
    Simulates potentiation-depression cycling without hardware.
    
    Args:
        pot_params (dict): Potentiation parameters
        dep_params (dict): Depression parameters
        n_cycles (int): Number of cycles
        update_callback (callable, optional): Function to call after each cycle
    
    Returns:
        dict: Results in same format as run_potentiation_depression_cycle
    """
    results = {
        "cycles": [],
        "pot_params": pot_params.copy(),
        "dep_params": dep_params.copy(),
        "n_cycles": n_cycles,
        "inter_cycle_delay_ms": 0
    }
    
    for cycle_idx in range(n_cycles):
        cycle_data = {
            "cycle_number": cycle_idx + 1,
            "potentiation": synapse_engine.simulate_pulse_read(pot_params),
            "depression": synapse_engine.simulate_pulse_read(dep_params)
        }
        results["cycles"].append(cycle_data)
        
        # Real-time update callback
        if update_callback is not None:
            try:
                update_callback(results)
            except Exception as e:
                print(f"⚠️ Update callback failed: {str(e)}")
        
        # Simulate some delay
        time.sleep(0.1)
    
    results["summary_metrics"] = calculate_cycle_summary_metrics(results)
    
    return results


# =============================================================================
# MULTI-DEVICE SEQUENTIAL TESTING
# =============================================================================

def run_multi_device_cycles(instrument, device_configs, update_callback=None):
    """
    Runs potentiation-depression cycles across multiple devices sequentially.
    
    Args:
        instrument: PyVISA instrument resource
        device_configs (list): List of device configuration dicts, each containing:
            - device_name (str): Identifier for the device
            - stim_ch (str): Stimulus channel
            - read_ch (str): Read channel
            - pot_params (dict): Potentiation parameters
            - dep_params (dict): Depression parameters
            - n_cycles (int): Number of cycles for this device
            - inter_cycle_delay_ms (float): Delay between cycles
        update_callback (callable, optional): Function called after each device with all results
    
    Returns:
        dict: Results containing:
            - devices: List of per-device results
            - device_names: List of device names tested
            - n_devices: Number of devices tested
    """
    multi_results = {
        "devices": [],
        "device_names": [],
        "n_devices": len(device_configs)
    }
    
    for device_idx, config in enumerate(device_configs):
        device_name = config.get("device_name", f"Device_{device_idx + 1}")
        
        print(f"\n{'#'*60}")
        print(f"# Testing Device: {device_name} ({device_idx + 1}/{len(device_configs)})")
        print(f"{'#'*60}")
        
        try:
            # Run cycles for this device
            device_results = run_potentiation_depression_cycle(
                instrument,
                config["stim_ch"],
                config["read_ch"],
                config["pot_params"],
                config["dep_params"],
                config["n_cycles"],
                config.get("inter_cycle_delay_ms", 1000),
                update_callback=None  # Don't update during individual cycles
            )
            
            # Add device identifier
            device_results["device_name"] = device_name
            multi_results["devices"].append(device_results)
            multi_results["device_names"].append(device_name)
            
            print(f"\n✓ Device {device_name} completed successfully")
            
            # Call update callback with all results so far
            if update_callback is not None:
                try:
                    update_callback(multi_results)
                except Exception as e:
                    print(f"⚠️ Update callback failed: {str(e)}")
            
            # Delay before next device (optional)
            if device_idx < len(device_configs) - 1:
                inter_device_delay = config.get("inter_device_delay_ms", 2000)
                print(f"\nWaiting {inter_device_delay} ms before next device...")
                time.sleep(inter_device_delay / 1000.0)
                
        except Exception as e:
            print(f"✗ Device {device_name} failed: {str(e)}")
            error_result = {
                "device_name": device_name,
                "error": str(e),
                "cycles": [],
                "n_cycles": 0
            }
            multi_results["devices"].append(error_result)
            multi_results["device_names"].append(device_name)
    
    return multi_results


def simulate_multi_device_cycles(device_configs, update_callback=None):
    """
    Simulates multi-device cycling without hardware.
    
    Args:
        device_configs (list): List of device configurations
        update_callback (callable, optional): Update function
    
    Returns:
        dict: Multi-device results
    """
    multi_results = {
        "devices": [],
        "device_names": [],
        "n_devices": len(device_configs)
    }
    
    for device_idx, config in enumerate(device_configs):
        device_name = config.get("device_name", f"Device_{device_idx + 1}")
        
        device_results = simulate_potentiation_depression_cycle(
            config["pot_params"],
            config["dep_params"],
            config["n_cycles"],
            update_callback=None
        )
        
        device_results["device_name"] = device_name
        multi_results["devices"].append(device_results)
        multi_results["device_names"].append(device_name)
        
        if update_callback is not None:
            try:
                update_callback(multi_results)
            except Exception as e:
                print(f"⚠️ Update callback failed: {str(e)}")
    
    return multi_results


# =============================================================================
# METRICS AND ANALYSIS
# =============================================================================

def calculate_cycle_summary_metrics(cycle_results):
    """
    Calculates aggregate metrics across all cycles.
    
    Args:
        cycle_results (dict): Results from run_potentiation_depression_cycle
    
    Returns:
        dict: Summary metrics including:
            - pot_delta_g_mean (%): Mean potentiation ΔG across cycles
            - pot_delta_g_std (%): Std dev of potentiation ΔG
            - dep_delta_g_mean (%): Mean depression ΔG across cycles
            - dep_delta_g_std (%): Std dev of depression ΔG
            - pot_reproducibility (%): Coefficient of variation for potentiation
            - dep_reproducibility (%): Coefficient of variation for depression
            - dynamic_range (S): Average difference between max pot and min dep conductance
    """
    metrics = {}
    
    pot_delta_g_list = []
    dep_delta_g_list = []
    pot_g_final_list = []
    dep_g_final_list = []
    
    for cycle in cycle_results["cycles"]:
        # Potentiation metrics
        if cycle["potentiation"] and "error" not in cycle["potentiation"]:
            pot_data = cycle["potentiation"]
            g_pot = np.array(pot_data["conductance_S"])
            g_pot_valid = g_pot[~np.isnan(g_pot)]
            
            if len(g_pot_valid) >= 2:
                delta_g_pot = (g_pot_valid[-1] - g_pot_valid[0]) / g_pot_valid[0] * 100
                pot_delta_g_list.append(delta_g_pot)
                pot_g_final_list.append(g_pot_valid[-1])
        
        # Depression metrics
        if cycle["depression"] and "error" not in cycle["depression"]:
            dep_data = cycle["depression"]
            g_dep = np.array(dep_data["conductance_S"])
            g_dep_valid = g_dep[~np.isnan(g_dep)]
            
            if len(g_dep_valid) >= 2:
                delta_g_dep = (g_dep_valid[-1] - g_dep_valid[0]) / g_dep_valid[0] * 100
                dep_delta_g_list.append(delta_g_dep)
                dep_g_final_list.append(g_dep_valid[-1])
    
    # Potentiation statistics
    if len(pot_delta_g_list) == 0:
        metrics["pot_delta_g_mean (%)"] = "N/A"
        metrics["pot_delta_g_std (%)"] = "N/A"
        metrics["pot_reproducibility_CV (%)"] = "N/A"
    else:
        metrics["pot_delta_g_mean (%)"] = round(np.mean(pot_delta_g_list), 2)
        metrics["pot_delta_g_std (%)"] = round(np.std(pot_delta_g_list), 2)
        
        # Reproducibility (coefficient of variation)
        if np.mean(pot_delta_g_list) != 0:
            cv_pot = (np.std(pot_delta_g_list) / abs(np.mean(pot_delta_g_list))) * 100
            metrics["pot_reproducibility_CV (%)"] = round(cv_pot, 2)
    
    # Depression statistics
    if len(dep_delta_g_list) == 0:
        metrics["dep_delta_g_mean (%)"] = "N/A"
        metrics["dep_delta_g_std (%)"] = "N/A"
        metrics["dep_reproducibility_CV (%)"] = "N/A"
    else:
        metrics["dep_delta_g_mean (%)"] = round(np.mean(dep_delta_g_list), 2)
        metrics["dep_delta_g_std (%)"] = round(np.std(dep_delta_g_list), 2)
        
        if np.mean(dep_delta_g_list) != 0:
            cv_dep = (np.std(dep_delta_g_list) / abs(np.mean(dep_delta_g_list))) * 100
            metrics["dep_reproducibility_CV (%)"] = round(cv_dep, 2)
    
    # Dynamic range
    if len(pot_g_final_list) > 0 and len(dep_g_final_list) > 0:
        avg_pot_g = np.mean(pot_g_final_list)
        avg_dep_g = np.mean(dep_g_final_list)
        dynamic_range = avg_pot_g - avg_dep_g
        metrics["dynamic_range (S)"] = f"{dynamic_range:.4e}"
        
        # On/Off ratio
        if avg_dep_g != 0:
            on_off_ratio = avg_pot_g / avg_dep_g
            metrics["on_off_ratio"] = round(on_off_ratio, 2)
    
    # Cycle-to-cycle variation
    if len(pot_delta_g_list) > 1:
        metrics["n_successful_cycles"] = len(pot_delta_g_list)
    
    return metrics


# =============================================================================
# DATA SAVING
# =============================================================================

def save_cycle_data(cycle_results, filename):
    """
    Saves potentiation-depression cycle data to CSV with metadata.
    
    Format:
        - Metadata header with parameters
        - Summary metrics
        - Per-cycle data with separate columns for pot/dep
    
    Args:
        cycle_results (dict): Results from run_potentiation_depression_cycle
        filename (str): Path to save CSV file
    
    Returns:
        None
    """
    csv_content = []
    
    # === METADATA SECTION ===
    csv_content.append("# Potentiation-Depression Cycle Characterization")
    csv_content.append(f"# n_cycles={cycle_results['n_cycles']}")
    csv_content.append(f"# inter_cycle_delay_ms={cycle_results.get('inter_cycle_delay_ms', 0)}")
    
    # Potentiation parameters
    pot_params = cycle_results["pot_params"]
    pot_param_str = " # ".join(f"pot_{k}={v}" for k, v in pot_params.items())
    csv_content.append(f"# {pot_param_str}")
    
    # Depression parameters
    dep_params = cycle_results["dep_params"]
    dep_param_str = " # ".join(f"dep_{k}={v}" for k, v in dep_params.items())
    csv_content.append(f"# {dep_param_str}")
    
    # Summary metrics
    summary = cycle_results.get("summary_metrics", {})
    if summary:
        summary_str = " # ".join(f"{k}={v}" for k, v in summary.items())
        csv_content.append(f"# SUMMARY: {summary_str}")
    
    csv_content.append("#")
    
    # === DATA SECTION ===
    # Column headers
    headers = [
        "cycle",
        "phase",  # 'potentiation' or 'depression'
        "pulse",
        "timestamp_s",
        "I_A",
        "V_read_V",
        "conductance_S"
    ]
    csv_content.append(",".join(headers))
    
    # Data rows
    for cycle in cycle_results["cycles"]:
        cycle_num = cycle["cycle_number"]
        
        # Potentiation data
        if cycle["potentiation"] and "error" not in cycle["potentiation"]:
            pot_data = cycle["potentiation"]
            for i in range(len(pot_data["pulse_number"])):
                row = [
                    str(cycle_num),
                    "potentiation",
                    str(pot_data["pulse_number"][i]),
                    f"{pot_data['time_s'][i]:.6f}",
                    f"{pot_data['I_A'][i]:.6e}",
                    f"{pot_data['V_read_V'][i]:.4f}",
                    f"{pot_data['conductance_S'][i]:.6e}"
                ]
                csv_content.append(",".join(row))
        
        # Depression data
        if cycle["depression"] and "error" not in cycle["depression"]:
            dep_data = cycle["depression"]
            for i in range(len(dep_data["pulse_number"])):
                row = [
                    str(cycle_num),
                    "depression",
                    str(dep_data["pulse_number"][i]),
                    f"{dep_data['time_s'][i]:.6f}",
                    f"{dep_data['I_A'][i]:.6e}",
                    f"{dep_data['V_read_V'][i]:.4f}",
                    f"{dep_data['conductance_S'][i]:.6e}"
                ]
                csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(csv_content))
    
    print(f"✓ Cycle data saved to {filename}")


def save_multi_device_data(multi_results, filename):
    """
    Saves multi-device cycle data to CSV.
    
    Args:
        multi_results (dict): Results from run_multi_device_cycles
        filename (str): Path to save CSV file
    """
    csv_content = []
    
    # Metadata
    csv_content.append("# Multi-Device Potentiation-Depression Cycle Characterization")
    csv_content.append(f"# n_devices={multi_results['n_devices']}")
    csv_content.append(f"# devices={','.join(multi_results['device_names'])}")
    csv_content.append("#")
    
    # Column headers
    headers = [
        "device",
        "cycle",
        "phase",
        "pulse",
        "timestamp_s",
        "I_A",
        "V_read_V",
        "conductance_S"
    ]
    csv_content.append(",".join(headers))
    
    # Data rows for each device
    for device_result in multi_results["devices"]:
        if "error" in device_result:
            continue
            
        device_name = device_result["device_name"]
        
        for cycle in device_result["cycles"]:
            cycle_num = cycle["cycle_number"]
            
            # Potentiation
            if cycle["potentiation"] and "error" not in cycle["potentiation"]:
                pot_data = cycle["potentiation"]
                for i in range(len(pot_data["pulse_number"])):
                    row = [
                        device_name,
                        str(cycle_num),
                        "potentiation",
                        str(pot_data["pulse_number"][i]),
                        f"{pot_data['time_s'][i]:.6f}",
                        f"{pot_data['I_A'][i]:.6e}",
                        f"{pot_data['V_read_V'][i]:.4f}",
                        f"{pot_data['conductance_S'][i]:.6e}"
                    ]
                    csv_content.append(",".join(row))
            
            # Depression
            if cycle["depression"] and "error" not in cycle["depression"]:
                dep_data = cycle["depression"]
                for i in range(len(dep_data["pulse_number"])):
                    row = [
                        device_name,
                        str(cycle_num),
                        "depression",
                        str(dep_data["pulse_number"][i]),
                        f"{dep_data['time_s'][i]:.6f}",
                        f"{dep_data['I_A'][i]:.6e}",
                        f"{dep_data['V_read_V'][i]:.4f}",
                        f"{dep_data['conductance_S'][i]:.6e}"
                    ]
                    csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(csv_content))
    
    print(f"✓ Multi-device data saved to {filename}")


# =============================================================================
# PARAMETER PRESETS
# =============================================================================

def get_preset_parameters(preset_name):
    """
    Returns preset parameter sets for common cycling protocols.
    
    Args:
        preset_name (str): One of:
            - "Standard Cycle": Balanced moderate protocol
            - "Fast Cycle": Quick cycling for screening
            - "High Endurance": Long-term stability testing
            - "Asymmetric": Different pot/dep parameters
    
    Returns:
        tuple: (pot_params, dep_params, n_cycles, inter_cycle_delay_ms)
    """
    presets = {
        "Standard Cycle": {
            "pot_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": 1.0,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 50,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "settle_ms": 5,
                "measure_avg": 1
            },
            "dep_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": -1.0,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 50,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "settle_ms": 5,
                "measure_avg": 1
            },
            "n_cycles": 5,
            "inter_cycle_delay_ms": 1000
        },
        
        "Fast Cycle": {
            "pot_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": 1.2,
                "stim_width_ms": 50,
                "stim_period_ms": 100,
                "n_pulses": 20,
                "read_voltage": 0.1,
                "read_delay_ms": 5,
                "compliance_A": 100e-6,
                "settle_ms": 3,
                "measure_avg": 1
            },
            "dep_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": -1.2,
                "stim_width_ms": 50,
                "stim_period_ms": 100,
                "n_pulses": 20,
                "read_voltage": 0.1,
                "read_delay_ms": 5,
                "compliance_A": 100e-6,
                "settle_ms": 3,
                "measure_avg": 1
            },
            "n_cycles": 10,
            "inter_cycle_delay_ms": 500
        },
        
        "High Endurance": {
            "pot_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": 0.8,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 100,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "settle_ms": 5,
                "measure_avg": 1
            },
            "dep_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": -0.8,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 100,
                "read_voltage": 0.1,
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "settle_ms": 5,
                "measure_avg": 1
            },
            "n_cycles": 20,
            "inter_cycle_delay_ms": 2000
        },
        
        "Asymmetric": {
            "pot_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": 1.5,
                "stim_width_ms": 50,
                "stim_period_ms": 150,
                "n_pulses": 30,
                "read_voltage": 0.15,  # Different read voltage
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "settle_ms": 5,
                "measure_avg": 1
            },
            "dep_params": {
                "mode": "electrical",
                "stim_drive_type": "V",
                "stim_level": -1.0,
                "stim_width_ms": 100,
                "stim_period_ms": 200,
                "n_pulses": 50,
                "read_voltage": 0.08,  # Different read voltage
                "read_delay_ms": 10,
                "compliance_A": 100e-6,
                "settle_ms": 5,
                "measure_avg": 1
            },
            "n_cycles": 5,
            "inter_cycle_delay_ms": 1500
        }
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = presets[preset_name]
    return (
        preset["pot_params"],
        preset["dep_params"],
        preset["n_cycles"],
        preset["inter_cycle_delay_ms"]
    )


def get_default_pot_params():
    """Returns default potentiation parameters."""
    return {
        "mode": "electrical",
        "stim_drive_type": "V",
        "stim_level": 1.0,
        "stim_width_ms": 100,
        "stim_period_ms": 200,
        "n_pulses": 50,
        "read_voltage": 0.1,
        "read_delay_ms": 10,
        "compliance_A": 100e-6,
        "settle_ms": 5,
        "measure_avg": 1
    }


def get_default_dep_params():
    """Returns default depression parameters."""
    return {
        "mode": "electrical",
        "stim_drive_type": "V",
        "stim_level": -1.0,
        "stim_width_ms": 100,
        "stim_period_ms": 200,
        "n_pulses": 50,
        "read_voltage": 0.1,
        "read_delay_ms": 10,
        "compliance_A": 100e-6,
        "settle_ms": 5,
        "measure_avg": 1
    }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    """Test the module with simulation"""
    print("Testing synapse_cycle module...")
    
    # Test preset loading
    print("\n=== Testing Presets ===")
    for preset_name in ["Standard Cycle", "Fast Cycle", "High Endurance", "Asymmetric"]:
        pot, dep, n_cyc, delay = get_preset_parameters(preset_name)
        print(f"{preset_name}: {n_cyc} cycles, pot={pot['stim_level']}V, dep={dep['stim_level']}V")
    
    # Test single-device cycling
    print("\n=== Testing Single Device ===")
    pot_params, dep_params, n_cycles, delay = get_preset_parameters("Fast Cycle")
    
    def test_callback(results):
        print(f"  Update: Completed {len(results['cycles'])}/{results['n_cycles']} cycles")
    
    results = simulate_potentiation_depression_cycle(
        pot_params, dep_params, n_cycles=3, update_callback=test_callback
    )
    
    print("\nSummary Metrics:")
    for key, value in results["summary_metrics"].items():
        print(f"  {key}: {value}")
    
    # Test multi-device
    print("\n=== Testing Multi-Device ===")
    device_configs = [
        {
            "device_name": "Device_A",
            "stim_ch": "smua",
            "read_ch": "smub",
            "pot_params": pot_params,
            "dep_params": dep_params,
            "n_cycles": 2
        },
        {
            "device_name": "Device_B",
            "stim_ch": "smua",
            "read_ch": "smub",
            "pot_params": pot_params,
            "dep_params": dep_params,
            "n_cycles": 2
        }
    ]
    
    def multi_callback(results):
        print(f"  Completed {len(results['devices'])}/{results['n_devices']} devices")
    
    multi_results = simulate_multi_device_cycles(device_configs, update_callback=multi_callback)
    print(f"\n✓ Tested {multi_results['n_devices']} devices successfully")
    
    print("\n✓ All tests complete!")