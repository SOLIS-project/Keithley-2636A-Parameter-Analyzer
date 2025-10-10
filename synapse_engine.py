"""
Synapse Engine Module for Keithley Sourcemeter
Implements pulse-read sequences for electrical, visual, and memristor synapse characterization.
Phase 1: Core engine and helper functions
"""

import pyvisa
import time
import numpy as np
from tkinter import messagebox

# =============================================================================
# STEP 1: HELPER FUNCTIONS
# =============================================================================

def open_instrument(port_str, connection_type):
    """
    Opens the pyvisa instrument resource based on connection type.
    
    Args:
        port_str (str): Port address/identifier
        connection_type (str): One of "GPIB", "RS232", or "LAN"
    
    Returns:
        pyvisa.Resource: Opened instrument resource
    
    Raises:
        ValueError: If connection cannot be established
    """
    rm = pyvisa.ResourceManager()
    instrument = None
    
    if connection_type == "GPIB":
        instrument = rm.open_resource(f"GPIB::{port_str}::INSTR")
    elif connection_type == "RS232":
        instrument = rm.open_resource(
            port_str, 
            baud_rate=9600, 
            data_bits=8, 
            parity=pyvisa.constants.Parity.none, 
            stop_bits=pyvisa.constants.StopBits.one, 
            flow_control=pyvisa.constants.VI_ASRL_FLOW_NONE
        )
    elif connection_type == "LAN":
        instrument = rm.open_resource(f"TCPIP::{port_str}::INSTR")
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

    if instrument is None:
        raise ValueError("Could not establish communication with the instrument.")
    
    return instrument


def validate_float(entry_value, field_name):
    """
    Validates and converts entry to float.
    
    Args:
        entry_value: Value to validate
        field_name (str): Name of field for error messages
    
    Returns:
        float: Validated float value
    
    Raises:
        ValueError: If conversion fails
    """
    try:
        return float(entry_value)
    except ValueError:
        raise ValueError(f"Invalid value for {field_name}. Please enter a valid number.")


def safety_check(stim_level, max_safe_voltage=2.5):
    """
    Performs safety check on stimulus level.
    
    Args:
        stim_level (float): Stimulus level in V or A
        max_safe_voltage (float): Maximum safe voltage without confirmation
    
    Returns:
        bool: True if safe to proceed, False if user cancels
    """
    if abs(stim_level) > max_safe_voltage:
        return messagebox.askyesno(
            "Safety Warning", 
            f"Stimulus level ({stim_level} V) exceeds recommended safe value ({max_safe_voltage} V).\n"
            "This may damage sensitive devices.\n\n"
            "Do you want to proceed?"
        )
    return True


# =============================================================================
# STEP 3: CORE ENGINE IMPLEMENTATION
# =============================================================================

def pulse_read_sequence(instrument, stim_ch, read_ch, params):
    """
    Core engine for electrical, visual, and memristor pulse measurements.
    
    Performs a sequence of stimulus pulses followed by read measurements to 
    characterize synaptic devices.
    
    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Stimulus channel ('smua' or 'smub')
        read_ch (str): Read channel ('smua' or 'smub')
        params (dict): Measurement parameters containing:
            - mode (str): 'electrical', 'visual', or 'memristor_pulse'
            - stim_drive_type (str): 'V' or 'I'
            - stim_level (float): Stimulus amplitude in V or A
            - stim_width_ms (float): Pulse width in milliseconds
            - stim_period_ms (float): Pulse period in milliseconds
            - n_pulses (int): Number of pulses
            - read_voltage (float): Read bias voltage in V
            - read_delay_ms (float): Delay after pulse before read (ms)
            - compliance_A (float): Current compliance in A
            - settle_ms (float, optional): Extra settling time (default: 5)
            - measure_avg (int, optional): Number of measurements to average (default: 1)
    
    Returns:
        dict: Results containing:
            - pulse_number: List of pulse indices
            - time_s: List of timestamps in seconds
            - I_A: List of measured currents in A
            - V_read_V: List of read voltages in V
            - conductance_S: List of conductance values in S
            - params: Copy of input parameters
    """
    results = {
        "pulse_number": [], 
        "time_s": [], 
        "I_A": [], 
        "V_read_V": [], 
        "conductance_S": [], 
        "params": params.copy()
    }
    
    # Extract parameters with defaults
    settle_ms = params.get('settle_ms', 5)
    measure_avg = params.get('measure_avg', 1)
    
    # --- CONFIGURATION PHASE ---
    
    # 1. Configure Read Channel (for low-level DC bias and measurement)
    instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{read_ch}.source.limiti = {params['compliance_A']}")
    instrument.write(f"{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_ON")
    instrument.write(f"{read_ch}.measure.nplc = 0.1")  # Fast reading
    instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
    
    # 2. Configure Stimulus Channel
    if params['stim_drive_type'] == 'V':
        stim_func = f"{stim_ch}.OUTPUT_DCVOLTS"
    else:
        stim_func = f"{stim_ch}.OUTPUT_DCAMPS"
    
    instrument.write(f"{stim_ch}.source.func = {stim_func}")
    instrument.write(f"{stim_ch}.source.limiti = {params['compliance_A']}")
    instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
    
    # 3. Set Read Channel to idle state (0V)
    instrument.write(f"{read_ch}.source.levelv = 0")
    instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
    
    # --- MEASUREMENT LOOP ---
    
    t0 = time.time()
    
    try:
        for i in range(params['n_pulses']):
            pulse_start = time.time()
            
            # --- STIMULUS ON ---
            if params['stim_drive_type'] == 'V':
                instrument.write(f"{stim_ch}.source.levelv = {params['stim_level']}")
            else:
                instrument.write(f"{stim_ch}.source.leveli = {params['stim_level']}")
            
            instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_ON")
            
            # Wait for pulse duration
            time.sleep(params['stim_width_ms'] / 1000.0)
            
            # --- STIMULUS OFF ---
            instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
            
            # --- WAIT READ DELAY ---
            time.sleep(params['read_delay_ms'] / 1000.0)
            
            # --- APPLY READ BIAS AND MEASURE CURRENT ---
            instrument.write(f"{read_ch}.source.levelv = {params['read_voltage']}")
            time.sleep(settle_ms / 1000.0)  # Settling time
            
            # Average multiple measurements
            I_sum = 0.0
            for _ in range(measure_avg):
                I = float(instrument.query(f"print({read_ch}.measure.i())").strip())
                I_sum += I
            I_meas = I_sum / measure_avg
            
            # --- RETURN READ CHANNEL TO IDLE ---
            instrument.write(f"{read_ch}.source.levelv = 0")
            
            # --- STORE RESULTS ---
            t_rel = time.time() - t0
            results["pulse_number"].append(i)
            results["time_s"].append(t_rel)
            results["I_A"].append(I_meas)
            results["V_read_V"].append(params['read_voltage'])
            
            # Calculate conductance
            if params['read_voltage'] != 0:
                results["conductance_S"].append(I_meas / params['read_voltage'])
            else:
                results["conductance_S"].append(float('nan'))
            
            # --- WAIT REMAINDER OF PERIOD ---
            pulse_elapsed = time.time() - pulse_start
            remaining_time = (params['stim_period_ms'] / 1000.0) - pulse_elapsed
            
            if remaining_time > 0:
                time.sleep(remaining_time)
            elif remaining_time < -0.010:  # Warn if >10ms over budget
                print(f"⚠️  Pulse {i}: Period exceeded by {-remaining_time*1000:.1f} ms")
        
    finally:
        # Ensure all outputs are OFF
        instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
    
    return results


# =============================================================================
# STEP 4: SIMULATION ENGINE
# =============================================================================

def simulate_pulse_read(params):
    """
    Simulates a pulse-read sequence for testing without hardware.
    Models simple exponential conductance change for LTP/LTD behavior.
    
    Args:
        params (dict): Same parameters as pulse_read_sequence
    
    Returns:
        dict: Results in same format as pulse_read_sequence
    """
    results = {
        "pulse_number": [], 
        "time_s": [], 
        "I_A": [], 
        "V_read_V": [], 
        "conductance_S": [], 
        "params": params.copy()
    }
    
    # Simulation parameters
    G_initial = 1e-6  # 1 µS initial conductance
    G_max = 1e-4      # 100 µS maximum conductance
    G_min = 1e-7      # 0.1 µS minimum conductance
    step_size = 0.05  # Learning rate (5% change per pulse)
    
    G_current = G_initial
    t0 = time.time()
    
    for i in range(params['n_pulses']):
        # Apply conductance change based on stimulus polarity
        if params['stim_level'] > 0:
            # LTP (Potentiation): Move toward G_max
            G_current += (G_max - G_current) * step_size
        else:
            # LTD (Depression): Move toward G_min
            G_current += (G_min - G_current) * step_size
        
        # Clip to physical limits
        G_current = np.clip(G_current, G_min, G_max)
        
        # Apply small random noise (±2%)
        G_current *= (1 + np.random.uniform(-0.02, 0.02))
        
        # Calculate measured current
        V_read = params['read_voltage']
        I_meas = G_current * V_read
        
        # Add measurement noise
        I_meas *= (1 + np.random.uniform(-0.01, 0.01))
        
        # Store results
        t_rel = time.time() - t0
        results["pulse_number"].append(i)
        results["time_s"].append(t_rel)
        results["I_A"].append(I_meas)
        results["V_read_V"].append(V_read)
        results["conductance_S"].append(G_current)
        
        # Simulate timing
        time.sleep(params['stim_period_ms'] / 1000.0)
    
    return results


# =============================================================================
# STEP 5: METRICS CALCULATION AND DATA SAVING
# =============================================================================

def calculate_synapse_metrics(results):
    """
    Calculates derived synaptic metrics from pulse-read results.
    
    Args:
        results (dict): Results from pulse_read_sequence or simulate_pulse_read
    
    Returns:
        dict: Calculated metrics including:
            - PPF (%): Paired-pulse facilitation
            - Mean Conductance (S): Average conductance
            - Delta G (S): Change in conductance (final - initial)
            - Delta G (%): Relative change in conductance
            - Max Conductance (S): Maximum observed conductance
            - Min Conductance (S): Minimum observed conductance
    """
    metrics = {}
    
    G_list = np.array(results["conductance_S"])
    I_list = np.array(results["I_A"])
    
    # Filter out NaN values
    G_valid = G_list[~np.isnan(G_list)]
    
    if len(G_valid) >= 2:
        # Paired-Pulse Facilitation (PPF) - ratio of 2nd to 1st pulse
        if G_valid[0] != 0:
            ppf = G_valid[1] / G_valid[0]
            metrics["PPF (%)"] = round(ppf * 100, 2)
        else:
            metrics["PPF (%)"] = "N/A"
        
        # Delta G (absolute and relative)
        delta_G = G_valid[-1] - G_valid[0]
        metrics["Delta G (S)"] = f"{delta_G:.4e}"
        
        if G_valid[0] != 0:
            delta_G_percent = (delta_G / G_valid[0]) * 100
            metrics["Delta G (%)"] = round(delta_G_percent, 2)
        else:
            metrics["Delta G (%)"] = "N/A"
    
    # Statistical measures
    if len(G_valid) > 0:
        metrics["Mean Conductance (S)"] = f"{np.mean(G_valid):.4e}"
        metrics["Max Conductance (S)"] = f"{np.max(G_valid):.4e}"
        metrics["Min Conductance (S)"] = f"{np.min(G_valid):.4e}"
        metrics["Std Conductance (S)"] = f"{np.std(G_valid):.4e}"
    
    # Current statistics
    if len(I_list) > 0:
        metrics["Mean Current (A)"] = f"{np.mean(I_list):.4e}"
        metrics["Max Current (A)"] = f"{np.max(I_list):.4e}"
        metrics["Min Current (A)"] = f"{np.min(I_list):.4e}"
    
    return metrics


def process_and_save_synapse_data(results_dict, filename):
    """
    Calculates derived metrics and saves data to CSV with metadata.
    
    Args:
        results_dict (dict): Results from pulse_read_sequence
        filename (str): Path to save CSV file
    
    Returns:
        dict: Calculated metrics
    """
    # Calculate metrics
    metrics = calculate_synapse_metrics(results_dict)
    
    # Prepare CSV content
    params = results_dict["params"]
    csv_content = []
    
    # Metadata header (key=value format)
    metadata_line = "# " + " # ".join(f"{k}={v}" for k, v in params.items())
    csv_content.append(metadata_line)
    
    # Add calculated metrics as metadata
    metrics_line = "# " + " # ".join(f"{k}={v}" for k, v in metrics.items())
    csv_content.append(metrics_line)
    
    # Column headers
    csv_content.append("pulse,timestamp_s,I_A,V_read_V,conductance_S")
    
    # Data rows
    for i in range(len(results_dict["pulse_number"])):
        row = [
            str(results_dict["pulse_number"][i]),
            f"{results_dict['time_s'][i]:.6f}",
            f"{results_dict['I_A'][i]:.6e}",
            f"{results_dict['V_read_V'][i]:.4f}",
            f"{results_dict['conductance_S'][i]:.6e}"
        ]
        csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(csv_content))
    
    return metrics



# =============================================================================
# STEP 6: SRDP AND STDP CHARACTERIZATION
# =============================================================================

def measure_srdp(instrument, stim_ch, read_ch, base_params, freq_list_hz):
    """
    Measures Spike-Rate-Dependent Plasticity (SRDP).
    
    Sweeps through different spike frequencies and measures the resulting
    conductance change (ΔG) to characterize rate-dependent learning.
    
    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Stimulus channel ('smua' or 'smub')
        read_ch (str): Read channel ('smua' or 'smub')
        base_params (dict): Base measurement parameters (stim_level, read_voltage, etc.)
        freq_list_hz (list): List of frequencies to test in Hz
    
    Returns:
        dict: SRDP results containing:
            - frequencies_hz: List of tested frequencies
            - delta_g_S: List of conductance changes
            - delta_g_percent: List of relative conductance changes
            - g_initial_S: List of initial conductances
            - g_final_S: List of final conductances
            - base_params: Copy of base parameters
    """
    results = {
        "frequencies_hz": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }
    
    for freq_hz in freq_list_hz:
        # Calculate period from frequency
        period_ms = 1000.0 / freq_hz
        
        # Ensure period is long enough for pulse width + read delay
        min_period = base_params['stim_width_ms'] + base_params['read_delay_ms'] + 10
        if period_ms < min_period:
            print(f"Warning: Frequency {freq_hz} Hz too high. Period {period_ms:.2f} ms < minimum {min_period:.2f} ms. Skipping.")
            continue
        
        # Update parameters for this frequency
        freq_params = base_params.copy()
        freq_params['stim_period_ms'] = period_ms
        
        # Run pulse-read sequence
        pulse_results = pulse_read_sequence(instrument, stim_ch, read_ch, freq_params)
        
        # Extract initial and final conductance
        g_list = np.array(pulse_results["conductance_S"])
        g_valid = g_list[~np.isnan(g_list)]
        
        if len(g_valid) >= 2:
            g_initial = g_valid[0]
            g_final = g_valid[-1]
            delta_g = g_final - g_initial
            
            if g_initial != 0:
                delta_g_percent = (delta_g / g_initial) * 100
            else:
                delta_g_percent = float('nan')
            
            # Store results
            results["frequencies_hz"].append(freq_hz)
            results["delta_g_S"].append(delta_g)
            results["delta_g_percent"].append(delta_g_percent)
            results["g_initial_S"].append(g_initial)
            results["g_final_S"].append(g_final)
        
        # Small delay between frequency measurements
        time.sleep(0.5)
    
    return results


def measure_stdp(instrument, stim_ch, read_ch, base_params, delta_t_list_ms):
    """
    Measures Spike-Timing-Dependent Plasticity (STDP).
    
    Sweeps through different pre-post spike time differences (Δt) and measures
    the resulting conductance change to characterize timing-dependent learning.
    
    Convention:
        - Δt > 0: Pre-spike before post-spike → typically LTP
        - Δt < 0: Post-spike before pre-spike → typically LTD
    
    Implementation:
        - Pre-spike: Applied on stim_ch
        - Post-spike: Applied on read_ch (dual role: post-spike + readout)
        - Δt is controlled by delay between pre and post pulses
    
    Args:
        instrument: PyVISA instrument resource
        stim_ch (str): Pre-synaptic stimulus channel ('smua' or 'smub')
        read_ch (str): Post-synaptic channel ('smua' or 'smub')
        base_params (dict): Base measurement parameters
        delta_t_list_ms (list): List of Δt values to test in milliseconds
    
    Returns:
        dict: STDP results containing:
            - delta_t_ms: List of time differences
            - delta_g_S: List of conductance changes
            - delta_g_percent: List of relative conductance changes
            - g_initial_S: List of initial conductances
            - g_final_S: List of final conductances
            - base_params: Copy of base parameters
    """
    results = {
        "delta_t_ms": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }
    
    # Configuration phase
    instrument.write(f"{stim_ch}.source.func = {stim_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{stim_ch}.source.limiti = {base_params['compliance_A']}")
    instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
    
    instrument.write(f"{read_ch}.source.func = {read_ch}.OUTPUT_DCVOLTS")
    instrument.write(f"{read_ch}.source.limiti = {base_params['compliance_A']}")
    instrument.write(f"{read_ch}.measure.autorangei = {read_ch}.AUTORANGE_ON")
    instrument.write(f"{read_ch}.measure.nplc = 0.1")
    instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
    
    pulse_width_s = base_params['stim_width_ms'] / 1000.0
    settle_ms = base_params.get('settle_ms', 5)
    n_pairs = base_params.get('n_pulses', 50)  # Number of spike pairs
    pair_period_ms = base_params.get('stim_period_ms', 200)  # Time between pairs
    
    for delta_t_ms in delta_t_list_ms:
        delta_t_s = delta_t_ms / 1000.0
        
        # Measure initial conductance
        instrument.write(f"{read_ch}.source.levelv = {base_params['read_voltage']}")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
        time.sleep(settle_ms / 1000.0)
        I_initial = float(instrument.query(f"print({read_ch}.measure.i())").strip())
        g_initial = I_initial / base_params['read_voltage'] if base_params['read_voltage'] != 0 else 0
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        
        # Apply spike pairs
        for pair_idx in range(n_pairs):
            if delta_t_ms >= 0:
                # Pre before Post (LTP): Δt > 0
                # Apply pre-spike
                instrument.write(f"{stim_ch}.source.levelv = {base_params['stim_level']}")
                instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_ON")
                time.sleep(pulse_width_s)
                instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
                
                # Wait Δt
                if delta_t_s > 0:
                    time.sleep(delta_t_s)
                
                # Apply post-spike
                post_level = base_params.get('post_spike_level', base_params['stim_level'])
                instrument.write(f"{read_ch}.source.levelv = {post_level}")
                instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
                time.sleep(pulse_width_s)
                instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
                
            else:
                # Post before Pre (LTD): Δt < 0
                # Apply post-spike
                post_level = base_params.get('post_spike_level', base_params['stim_level'])
                instrument.write(f"{read_ch}.source.levelv = {post_level}")
                instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
                time.sleep(pulse_width_s)
                instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
                
                # Wait |Δt|
                time.sleep(abs(delta_t_s))
                
                # Apply pre-spike
                instrument.write(f"{stim_ch}.source.levelv = {base_params['stim_level']}")
                instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_ON")
                time.sleep(pulse_width_s)
                instrument.write(f"{stim_ch}.source.output = {stim_ch}.OUTPUT_OFF")
            
            # Wait for next pair
            time.sleep(pair_period_ms / 1000.0)
        
        # Measure final conductance
        instrument.write(f"{read_ch}.source.levelv = {base_params['read_voltage']}")
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_ON")
        time.sleep(settle_ms / 1000.0)
        I_final = float(instrument.query(f"print({read_ch}.measure.i())").strip())
        g_final = I_final / base_params['read_voltage'] if base_params['read_voltage'] != 0 else 0
        instrument.write(f"{read_ch}.source.output = {read_ch}.OUTPUT_OFF")
        
        # Calculate change
        delta_g = g_final - g_initial
        delta_g_percent = (delta_g / g_initial) * 100 if g_initial != 0 else float('nan')
        
        # Store results
        results["delta_t_ms"].append(delta_t_ms)
        results["delta_g_S"].append(delta_g)
        results["delta_g_percent"].append(delta_g_percent)
        results["g_initial_S"].append(g_initial)
        results["g_final_S"].append(g_final)
        
        # Recovery time between measurements
        time.sleep(1.0)
    
    return results


def simulate_srdp(base_params, freq_list_hz):
    """
    Simulates SRDP behavior for testing without hardware.
    Models frequency-dependent conductance change.
    """
    results = {
        "frequencies_hz": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }
    
    G_initial = 1e-6  # 1 µS
    
    for freq_hz in freq_list_hz:
        # Model: Higher frequency → stronger potentiation
        # Logarithmic relationship with saturation
        freq_factor = np.log10(freq_hz + 1) / np.log10(100)  # Normalize to 0-1 range
        
        if base_params['stim_level'] > 0:
            # LTP: Positive correlation with frequency
            delta_g_normalized = freq_factor * 0.5  # Up to 50% change
        else:
            # LTD: Negative correlation with frequency
            delta_g_normalized = -freq_factor * 0.3  # Up to -30% change
        
        g_final = G_initial * (1 + delta_g_normalized)
        delta_g = g_final - G_initial
        delta_g_percent = delta_g_normalized * 100
        
        # Add noise
        g_final *= (1 + np.random.uniform(-0.05, 0.05))
        delta_g = g_final - G_initial
        
        results["frequencies_hz"].append(freq_hz)
        results["delta_g_S"].append(delta_g)
        results["delta_g_percent"].append(delta_g_percent)
        results["g_initial_S"].append(G_initial)
        results["g_final_S"].append(g_final)
    
    return results


def simulate_stdp(base_params, delta_t_list_ms):
    """
    Simulates STDP behavior for testing without hardware.
    Models classic exponential STDP window.
    """
    results = {
        "delta_t_ms": [],
        "delta_g_S": [],
        "delta_g_percent": [],
        "g_initial_S": [],
        "g_final_S": [],
        "base_params": base_params.copy()
    }
    
    G_initial = 1e-6  # 1 µS
    A_plus = 0.5   # LTP amplitude
    A_minus = 0.3  # LTD amplitude
    tau_plus = 20  # LTP time constant (ms)
    tau_minus = 20 # LTD time constant (ms)
    
    for delta_t_ms in delta_t_list_ms:
        if delta_t_ms > 0:
            # Pre before Post → LTP
            delta_g_normalized = A_plus * np.exp(-delta_t_ms / tau_plus)
        else:
            # Post before Pre → LTD
            delta_g_normalized = -A_minus * np.exp(delta_t_ms / tau_minus)
        
        g_final = G_initial * (1 + delta_g_normalized)
        delta_g = g_final - G_initial
        delta_g_percent = delta_g_normalized * 100
        
        # Add noise
        g_final *= (1 + np.random.uniform(-0.05, 0.05))
        delta_g = g_final - G_initial
        
        results["delta_t_ms"].append(delta_t_ms)
        results["delta_g_S"].append(delta_g)
        results["delta_g_percent"].append(delta_g_percent)
        results["g_initial_S"].append(G_initial)
        results["g_final_S"].append(g_final)
    
    return results


# =============================================================================
# MODULE TEST (optional - can be removed in production)
# =============================================================================

if __name__ == "__main__":
    """Test the module with simulation"""
    print("Testing synapse_engine module...")
    
    # Test parameters
    test_params = {
        "mode": "electrical",
        "stim_drive_type": "V",
        "stim_level": 1.0,
        "stim_width_ms": 100,
        "stim_period_ms": 200,
        "n_pulses": 10,
        "read_voltage": 0.1,
        "read_delay_ms": 10,
        "compliance_A": 100e-6,
        "settle_ms": 5,
        "measure_avg": 1
    }
    
    # Run simulation
    print("\nRunning simulation...")
    results = simulate_pulse_read(test_params)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_synapse_metrics(results)
    
    # Display results
    print("\n=== Synapse Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save to file
    print("\nSaving to test_synapse_data.csv...")
    process_and_save_synapse_data(results, "test_synapse_data.csv")
    print("✓ Test complete!")