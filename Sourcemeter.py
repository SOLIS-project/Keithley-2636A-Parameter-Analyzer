import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyvisa
import time
from tkinter.filedialog import asksaveasfilename
import synapse_engine
import synapse_cycle

# appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Arrays to store JV curves and PV parameters
jv_curves = []  # To store [(sample_name, voltages, current_density)]
pv_parameters = []  # To store [(sample_name, parameters_dict)]
synapse_data_storage = []  # To store synapse measurement results

def calculate_pv_parameters(voltages, currents, area, light_power):
    """Calculate photovoltaic parameters from JV curve data."""
    current_density = np.array(currents) * 1e3 / area  # A to mA/cm²
    
    # Find the indices where current density changes sign
    pos_indices = np.where(current_density > 0)[0]
    neg_indices = np.where(current_density < 0)[0]
    
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        # Find the closest pair of points straddling J = 0
        idx_low = neg_indices[-1]  # Last negative current index
        idx_high = pos_indices[0]  # First positive current index
    
        # Perform linear interpolation to estimate Voc
        v_low, j_low = voltages[idx_low], current_density[idx_low]
        v_high, j_high = voltages[idx_high], current_density[idx_high]
        voc = v_low - j_low * (v_high - v_low) / (j_high - j_low)
    else:
        # Fallback if interpolation is not possible
        voc = voltages[np.argmin(np.abs(current_density))]  

    # Find Jsc (mA/cm²)
    jsc_index = np.argmin(np.abs(voltages))
    jsc = -current_density[jsc_index]
    
    # Calculate power density (mW/cm²)
    power_density = np.array(voltages) * current_density
    
    # Find maximum power point
    mpp_index = np.argmin(power_density)
    vmp = voltages[mpp_index]
    jmp = current_density[mpp_index]
    
    # Calculate Fill Factor (%)
    ff = np.abs(vmp * jmp / (voc * jsc)) * 100
    
    # Calculate PCE (%)
    light_power_mw_cm2 = light_power / 10
    pce = np.abs(power_density[mpp_index]) / light_power_mw_cm2 * 100
    
    return {
        "Voc": round(voc, 3),
        "Jsc": round(jsc, 3),
        "FF": round(ff, 2),
        "PCE": round(pce, 2)
    }

def update_parameters_display(params):

    params_text.insert("end", f"Cell # {params['Cell #']}:\n")
    params_text.insert("end", f"Voc: {params['Voc']} V\n")
    params_text.insert("end", f"Jsc: {params['Jsc']} mA/cm²\n")
    params_text.insert("end", f"FF: {params['FF']} %\n")
    params_text.insert("end", f"PCE: {params['PCE']} %\n")
    params_text.insert("end", "-"*20 + "\n")

def validate_float(entry_value, field_name):
    try:
        return float(entry_value)
    except ValueError:
        raise ValueError(f"Invalid value for {field_name}. Please enter a valid number.")

def run_measurement_buffered(transistor_mode=False):
    global voltages, current_density, jv_curves
    try:
        # Validate and parse GUI inputs
        sample_name = sample_name_entry.get().strip()
        if not sample_name:
            raise ValueError("Cell # cannot be empty.")

        area = validate_float(surface_area.get(), "Sample Surface Area")
        if area <= 0:
            raise ValueError("Surface area must be greater than zero.")

        start = validate_float(start_voltage.get(), "Starting Voltage")
        end = validate_float(end_voltage.get(), "Ending Voltage")
        step = validate_float(voltage_step.get(), "Voltage Step")
        nplc = validate_float(nplc_entry.get(), "Measurement Speed (NPLC)")

        compliance_str = compliance.get()
        try:
            compliance_value = float(compliance_str[:-2]) * 10**{"n": -9, "µ": -6, "m": -3, "A": 0}[compliance_str[-2]]
        except (ValueError, KeyError):
            raise ValueError("Invalid compliance value.")

        channel = channel_selection.get()
        connection = connection_type.get()
        port = port_entry.get().strip()
        if not port:
            raise ValueError("Port Address cannot be empty.")

        hysteresis_cycles_value = 1
        if hysteresis_var.get():
            try:
                hysteresis_cycles_value = int(hysteresis_cycles.get())
                if hysteresis_cycles_value <= 0:
                    raise ValueError("Hysteresis Cycles must be greater than zero.")
            except ValueError:
                raise ValueError("Invalid value for Hysteresis Cycles.")

        if not dark_measurement.get():
            light_power = validate_float(light_power_entry.get(), "Irradiance")
            if light_power <= 0:
                raise ValueError("Irradiance must be greater than zero.")

        # Initialize communication with Keithley 2636A
        rm = pyvisa.ResourceManager()
        instrument = None
        if connection == "GPIB":
            instrument = rm.open_resource(f"GPIB::{port}::INSTR")
        elif connection == "RS232":
            instrument = rm.open_resource(port, baud_rate=9600, data_bits=8, parity=pyvisa.constants.Parity.none, stop_bits=pyvisa.constants.StopBits.one, flow_control=pyvisa.constants.VI_ASRL_FLOW_NONE)
        elif connection == "LAN":
            instrument = rm.open_resource(f"TCPIP::{port}::INSTR")

        if instrument is None:
            raise ValueError("Could not establish communication with the instrument.")

        # Get user-specified timeout
        try:
            user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
            if user_timeout <= 0:
                raise ValueError("Timeout duration must be greater than zero.")
            timeout_value = user_timeout * 1000  # Convert seconds to milliseconds
        except ValueError:
            # Fallback to calculated timeout if invalid input
            timeout_value = (nplc * (1/60) * 3) * 1000
            timeout_value = max(5000, timeout_value)
        
        instrument.timeout = timeout_value

        # Instrument initialization
        if not transistor_mode:
            instrument.write("*RST")
        instrument.write("*CLS")
        channel_cmd = "smua" if channel == "Channel A" else "smub"

        # Configure source and measurement
        instrument.write(f"{channel_cmd}.source.func = {channel_cmd}.OUTPUT_DCVOLTS")
        instrument.write(f"{channel_cmd}.source.autorangev = {channel_cmd}.AUTORANGE_ON")
        if autorange_var.get():
            instrument.write(f"{channel_cmd}.measure.autorangei = {channel_cmd}.AUTORANGE_ON")
        else:
            instrument.write(f"{channel_cmd}.measure.autorangei = {channel_cmd}.AUTORANGE_OFF")

        # Update timeout if autorange is enabled (may need more time)
        if autorange_var.get():
            try:
                user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
                timeout_value = max(user_timeout * 1000, 10000)  # At least 10 seconds for autorange
            except ValueError:
                timeout_value = max(10000, nplc * (1/60) * 5 * 1000)
            instrument.timeout = timeout_value

        instrument.write(f"{channel_cmd}.source.limiti = {compliance_value}")
        instrument.write(f"{channel_cmd}.measure.nplc = {nplc}")
        instrument.write(f"{channel_cmd}.nvbuffer1.clear()")
        instrument.write(f"{channel_cmd}.nvbuffer1.appendmode = 1")
        instrument.write(f"{channel_cmd}.nvbuffer1.collectsourcevalues = 1")
        
        if wire_mode.get() == "4-Wire":
            instrument.write(f"{channel_cmd}.sense = {channel_cmd}.SENSE_REMOTE")
        else:
            instrument.write(f"{channel_cmd}.sense = {channel_cmd}.SENSE_LOCAL")

        # Define voltage sweep
        voltages = np.arange(start, end + step, step)
        
        if hysteresis_var.get():
            reverse_voltages = voltages[::-1]
            full_cycle_voltages = np.concatenate([voltages, reverse_voltages])
        else:
            full_cycle_voltages = voltages

        # Perform measurement
        all_currents = []
        measurement_delay = nplc * (1/60)
        for cycle in range(hysteresis_cycles_value):
            for voltage in full_cycle_voltages:
                instrument.write(f"{channel_cmd}.source.levelv = {voltage}")
                instrument.write(f"{channel_cmd}.source.output = {channel_cmd}.OUTPUT_ON")
                time.sleep(max(0.01, measurement_delay))
                instrument.write(f"{channel_cmd}.measure.i({channel_cmd}.nvbuffer1)")
                time.sleep(max(0.01, measurement_delay/2))
            instrument.write(f"{channel_cmd}.source.output = {channel_cmd}.OUTPUT_OFF")

        # Retrieve data
        currents = instrument.query_ascii_values(f"printbuffer(1, {len(full_cycle_voltages) * hysteresis_cycles_value}, {channel_cmd}.nvbuffer1.readings)")
        voltages = instrument.query_ascii_values(f"printbuffer(1, {len(full_cycle_voltages) * hysteresis_cycles_value}, {channel_cmd}.nvbuffer1.sourcevalues)")

        # Process and plot data
        current_density = np.array(currents) * 1e3 / area
        ax.clear()
        
        if hasattr(ax, 'ax2') and ax.ax2 is not None:
            ax.ax2.remove()
            ax.ax2 = None
        plt.cla()
        
        ax.plot(voltages, current_density, label="JV Curve", color="blue")
        ax.set_title("JV Curve")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density (mA/cm²)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        
        if not dark_measurement.get() and not hysteresis_var.get():
            power_density = np.array(voltages) * current_density
            pv_params = calculate_pv_parameters(
                voltages, 
                currents, 
                area,
                float(light_power_entry.get())
            )
            pv_params["Cell #"] = sample_name
            pv_parameters.append(pv_params)
            update_parameters_display(pv_params)
        
            ax2 = ax.twinx()
            ax2.plot(voltages, power_density, label="Power Density", color="red", linestyle="--")
            ax2.set_ylabel("Power Density (mW/cm²)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper right")
            ax.ax2 = ax2
        else:
            if hasattr(ax, 'ax2') and ax.ax2:
                ax.ax2.remove()
                ax.ax2 = None
        
        ax.legend(loc="upper left")
        canvas.draw()
        add_semilog_jv_curve(graph_frame, voltages, current_density)

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        if 'instrument' in locals() and instrument is not None:
            instrument.write(f"{channel_cmd}.source.output = {channel_cmd}.OUTPUT_OFF")
            instrument.close()

    jv_curves.append({"Cell #": sample_name, "Voltages": voltages, "Current Density": current_density, "Surface Area": area, "Measurement Type": "Dark" if dark_measurement.get() else "Illuminated"})

def save_curve():
    try:
        jv_file = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save JV Curves")
        if jv_file:
            header_sample_names = []
            header_labels = []
            data_rows = []
            
            max_length = max(len(entry["Voltages"]) for entry in jv_curves)
            
            for entry in jv_curves:
                sample_name = entry["Cell #"]
                if entry["Measurement Type"] == "Dark":
                    sample_name += " -- Dark"
                    
                if entry["Measurement Type"] == "Dark":
                    header_sample_names.extend([f"{sample_name}", f"{entry['Surface Area']}"])
                else:
                    irradiance_value = light_power_entry.get().strip()
                    header_sample_names.extend([f"{sample_name} ({irradiance_value} W/m²)", f"{entry['Surface Area']}"])

                header_labels.extend(["Voltage (V)", "Current Density (mA/cm²)"])
                
                voltages = entry["Voltages"]
                current_density = entry["Current Density"]
                for i in range(max_length):
                    if len(data_rows) <= i:
                        data_rows.append([])
                    data_rows[i].extend([
                        voltages[i] if i < len(voltages) else "",
                        current_density[i] if i < len(current_density) else ""
                    ])
            
            with open(jv_file, "w") as f:
                f.write(",".join(header_sample_names) + "\n")
                f.write(",".join(header_labels) + "\n")
                for row in data_rows:
                    f.write(",".join(map(str, row)) + "\n")

        pv_file = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save PV Parameters")
        if pv_file:
            with open(pv_file, "w") as f:
                irradiance_value = light_power_entry.get().strip()
                f.write("Cell # (Irradiance W/m²),Voc (V),Jsc (mA/cm²),FF (%),PCE (%)\n")
                for entry in pv_parameters:
                    f.write(f"{entry['Cell #']} ({irradiance_value} W/m²),{entry['Voc']},{entry['Jsc']},{entry['FF']},{entry['PCE']}\n")

        messagebox.showinfo("Save Successful", "Data saved successfully.")
        
        jv_curves.clear()
        pv_parameters.clear()
        params_text.delete("1.0", "end")
        
    except Exception as e:
        messagebox.showerror("Save Error", str(e))

def add_semilog_jv_curve(graph_frame, voltages, currents):

    if hasattr(graph_frame, "semilog_frame") and graph_frame.semilog_frame is not None:
        graph_frame.semilog_frame.destroy()

    graph_frame.semilog_frame = ctk.CTkFrame(graph_frame)
    graph_frame.semilog_frame.pack(fill="both", expand=True)

    fig_semi = plt.Figure(figsize=(5, 2))
    ax_semi = fig_semi.add_subplot(111)
    
    abs_currents = np.abs(currents)
    ax_semi.semilogy(voltages, abs_currents, label="JV Curve (Semilog)", color="green")
    
    ax_semi.set_title("JV Curve (Semilog)")
    ax_semi.set_xlabel("Voltage (V)")
    ax_semi.set_ylabel("Current Density (mA/cm²)")
    ax_semi.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_semi.legend()

    canvas_semi = FigureCanvasTkAgg(fig_semi, master=graph_frame.semilog_frame)
    canvas_semi_widget = canvas_semi.get_tk_widget()
    canvas_semi_widget.pack(fill="both", expand=True)

    return graph_frame.semilog_frame

def plot_synapse_data(results):
    """
    Plots synapse measurement results (Conductance vs Pulse Number).
    
    Args:
        results (dict): Results from synapse_engine.pulse_read_sequence
    """
    global ax, canvas
    
    # Clear existing plots
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    conductance = np.array(results["conductance_S"])
    pulse_numbers = results["pulse_number"]
    currents = np.array(results["I_A"])
    
    # Main plot: Conductance vs Pulse Number
    color = 'tab:blue'
    ax.plot(pulse_numbers, conductance * 1e6, 'o-', color=color, linewidth=2, markersize=4)
    ax.set_xlabel("Pulse Number", fontsize=11)
    ax.set_ylabel("Conductance (µS)", color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title(f"{results['params']['mode'].capitalize()} Synapse Response", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Secondary y-axis: Current
    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.plot(pulse_numbers, currents * 1e6, 's--', color=color, linewidth=1.5, markersize=3, alpha=0.7)
    ax2.set_ylabel("Current (µA)", color=color, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color)
    ax.ax2 = ax2
    
    # Add legend
    ax.legend(['Conductance'], loc='upper left')
    ax2.legend(['Current'], loc='upper right')
    
    canvas.draw()
    
    # Calculate and display metrics
    metrics = synapse_engine.calculate_synapse_metrics(results)
    
    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"Synapse Measurement Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Mode: {results['params']['mode']}\n")
    params_text.insert("end", f"Pulses: {results['params']['n_pulses']}\n")
    params_text.insert("end", f"Stim Level: {results['params']['stim_level']} {results['params']['stim_drive_type']}\n")
    params_text.insert("end", f"{'-'*30}\n")
    
    for key, value in metrics.items():
        params_text.insert("end", f"{key}: {value}\n")
    
    params_text.insert("end", f"{'='*30}\n\n")
    
    # Scroll to bottom
    params_text.see("end")



def plot_srdp_data(results):
    """
    Plots SRDP characterization results (ΔG vs Frequency).
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    frequencies = results["frequencies_hz"]
    delta_g_percent = results["delta_g_percent"]
    
    # Main plot: ΔG% vs Frequency
    color = 'tab:blue'
    ax.plot(frequencies, delta_g_percent, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_xlabel("Spike Frequency (Hz)", fontsize=11)
    ax.set_ylabel("ΔG (%)", color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title("Spike-Rate-Dependent Plasticity (SRDP)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    canvas.draw()
    
    # Display results in text box
    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"SRDP Characterization Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Stim Level: {results['base_params']['stim_level']} V\n")
    params_text.insert("end", f"Pulses per freq: {results['base_params']['n_pulses']}\n")
    params_text.insert("end", f"{'-'*30}\n")
    
    for i, freq in enumerate(frequencies):
        params_text.insert("end", 
            f"{freq:.1f} Hz → ΔG = {delta_g_percent[i]:.2f}%\n")
    
    params_text.insert("end", f"{'='*30}\n\n")
    params_text.see("end")


def plot_stdp_data(results):
    """
    Plots STDP characterization results (ΔG vs Δt).
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    delta_t = results["delta_t_ms"]
    delta_g_percent = results["delta_g_percent"]
    
    # Main plot: ΔG% vs Δt
    color = 'tab:green'
    ax.plot(delta_t, delta_g_percent, 'o-', color=color, linewidth=2, markersize=6)
    ax.set_xlabel("Δt (ms) [Pre - Post]", fontsize=11)
    ax.set_ylabel("ΔG (%)", color=color, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title("Spike-Timing-Dependent Plasticity (STDP)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    # Annotations
    ax.text(0.05, 0.95, 'LTP (Δt > 0)', transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', color='blue')
    ax.text(0.05, 0.05, 'LTD (Δt < 0)', transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', color='red')
    
    canvas.draw()
    
    # Display results in text box
    params_text.insert("end", f"\n{'='*30}\n")
    params_text.insert("end", f"STDP Characterization Results\n")
    params_text.insert("end", f"{'='*30}\n")
    params_text.insert("end", f"Stim Level: {results['base_params']['stim_level']} V\n")
    params_text.insert("end", f"Spike pairs: {results['base_params'].get('n_pulses', 50)}\n")
    params_text.insert("end", f"{'-'*30}\n")
    
    for i, dt in enumerate(delta_t):
        params_text.insert("end", 
            f"Δt = {dt:+.1f} ms → ΔG = {delta_g_percent[i]:+.2f}%\n")
    
    params_text.insert("end", f"{'='*30}\n\n")
    params_text.see("end")


def plot_cycle_data(results):
    """
    Plots potentiation-depression cycle results with real-time updates.
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    n_completed = len(results["cycles"])
    colors_pot = plt.cm.Reds(np.linspace(0.4, 0.9, max(n_completed, 1)))
    colors_dep = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_completed, 1)))
    
    max_pulse_count = 0
    
    for i, cycle in enumerate(results["cycles"]):
        cycle_num = cycle["cycle_number"]
        pulse_offset = 0
        
        # Plot potentiation
        if cycle["potentiation"] and "error" not in cycle["potentiation"]:
            pot_data = cycle["potentiation"]
            g_pot = np.array(pot_data["conductance_S"]) * 1e6  # Convert to µS
            pulse_num = pot_data["pulse_number"]
            ax.plot(pulse_num, g_pot, 'o-', color=colors_pot[i], 
                   linewidth=2, markersize=4, label=f'Cycle {cycle_num} Pot', alpha=0.8)
            pulse_offset = len(pulse_num)
            max_pulse_count = max(max_pulse_count, max(pulse_num))
        
        # Plot depression
        if cycle["depression"] and "error" not in cycle["depression"]:
            dep_data = cycle["depression"]
            g_dep = np.array(dep_data["conductance_S"]) * 1e6  # Convert to µS
            pulse_num = dep_data["pulse_number"]
            pulse_num_shifted = [p + pulse_offset for p in pulse_num]
            ax.plot(pulse_num_shifted, g_dep, 's--', color=colors_dep[i], 
                   linewidth=2, markersize=4, label=f'Cycle {cycle_num} Dep', alpha=0.8)
            max_pulse_count = max(max_pulse_count, max(pulse_num_shifted))
    
    ax.set_xlabel("Pulse Number", fontsize=11)
    ax.set_ylabel("Conductance (µS)", fontsize=11)
    
    # Update title with progress
    if n_completed < results['n_cycles']:
        title = f"Pot-Dep Cycles (In Progress: {n_completed}/{results['n_cycles']})"
    else:
        title = f"Pot-Dep Cycles (Complete: {n_completed}/{results['n_cycles']})"
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    canvas.draw()
    
    # Update text display (only show summary if complete)
    if n_completed == results['n_cycles'] and 'summary_metrics' in results:
        params_text.delete("1.0", "end")  # Clear previous text
        params_text.insert("end", f"\n{'='*30}\n")
        params_text.insert("end", f"Cycle Characterization Results\n")
        params_text.insert("end", f"{'='*30}\n")
        params_text.insert("end", f"Cycles completed: {results['n_cycles']}\n")
        params_text.insert("end", f"{'-'*30}\n")
        
        summary = results.get("summary_metrics", {})
        for key, value in summary.items():
            params_text.insert("end", f"{key}: {value}\n")
        
        params_text.insert("end", f"{'='*30}\n\n")
        params_text.see("end")


def plot_multi_device_data(multi_results):
    """
    Plots multi-device cycle results.
    """
    global ax, canvas
    
    ax.clear()
    if hasattr(ax, 'ax2') and ax.ax2 is not None:
        ax.ax2.remove()
        ax.ax2 = None
    
    device_colors = plt.cm.tab10(np.linspace(0, 1, multi_results['n_devices']))
    
    for dev_idx, device_result in enumerate(multi_results["devices"]):
        if "error" in device_result:
            continue
        
        device_name = device_result["device_name"]
        
        # Calculate average conductance trajectory for this device
        all_pot_g = []
        all_dep_g = []
        
        for cycle in device_result["cycles"]:
            if cycle["potentiation"] and "error" not in cycle["potentiation"]:
                g_pot = np.array(cycle["potentiation"]["conductance_S"]) * 1e6
                all_pot_g.append(g_pot)
            
            if cycle["depression"] and "error" not in cycle["depression"]:
                g_dep = np.array(cycle["depression"]["conductance_S"]) * 1e6
                all_dep_g.append(g_dep)
        
        if all_pot_g:
            avg_pot = np.mean(all_pot_g, axis=0)
            pulse_num = range(len(avg_pot))
            ax.plot(pulse_num, avg_pot, 'o-', color=device_colors[dev_idx],
                   linewidth=2, markersize=4, label=f'{device_name} Pot')
        
        if all_dep_g:
            avg_dep = np.mean(all_dep_g, axis=0)
            pulse_offset = len(avg_pot) if all_pot_g else 0
            pulse_num = [p + pulse_offset for p in range(len(avg_dep))]
            ax.plot(pulse_num, avg_dep, 's--', color=device_colors[dev_idx],
                   linewidth=2, markersize=4, label=f'{device_name} Dep')
    
    ax.set_xlabel("Pulse Number", fontsize=11)
    ax.set_ylabel("Conductance (µS)", fontsize=11)
    ax.set_title(f"Multi-Device Comparison ({multi_results['n_devices']} devices)", 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    canvas.draw()
    
    # Display summary for each device
    params_text.delete("1.0", "end")
    params_text.insert("end", f"\n{'='*40}\n")
    params_text.insert("end", f"Multi-Device Results\n")
    params_text.insert("end", f"{'='*40}\n")
    
    for device_result in multi_results["devices"]:
        if "error" in device_result:
            params_text.insert("end", f"\n{device_result['device_name']}: ERROR\n")
            params_text.insert("end", f"  {device_result['error']}\n")
            continue
        
        params_text.insert("end", f"\n{device_result['device_name']}:\n")
        summary = device_result.get("summary_metrics", {})
        for key, value in summary.items():
            params_text.insert("end", f"  {key}: {value}\n")
    
    params_text.insert("end", f"\n{'='*40}\n")
    params_text.see("end")








def run_srdp_characterization():
    """
    Runs SRDP characterization with frequency sweep.
    """
    global synapse_data_storage
    
    try:
        # Parse frequency range
        freq_start = validate_float(srdp_freq_start_entry.get(), "SRDP Start Frequency")
        freq_end = validate_float(srdp_freq_end_entry.get(), "SRDP End Frequency")
        freq_points = int(validate_float(srdp_freq_points_entry.get(), "SRDP Frequency Points"))
        
        if freq_points < 2:
            raise ValueError("Need at least 2 frequency points")
        
        # Generate frequency list (log scale for better coverage)
        if srdp_log_scale_var.get():
            freq_list = np.logspace(np.log10(freq_start), np.log10(freq_end), freq_points)
        else:
            freq_list = np.linspace(freq_start, freq_end, freq_points)
        
        # Check if frequency range is achievable with current timing parameters
        # Use SRDP-specific fields
        stim_width = validate_float(srdp_stim_width_entry.get(), "SRDP Stim Width")
        read_delay = validate_float(read_delay_entry.get(), "Read Delay")
        min_period = stim_width + read_delay + 10
        max_achievable_freq = 1000.0 / min_period
        
        if freq_end > max_achievable_freq:
            warning_msg = (
                f"⚠️ Frequency range issue:\n\n"
                f"Your timing parameters limit the maximum frequency:\n"
                f"  • Stim Width: {stim_width} ms\n"
                f"  • Read Delay: {read_delay} ms\n"
                f"  • Min Period: {min_period} ms\n"
                f"  • Max Frequency: {max_achievable_freq:.2f} Hz\n\n"
                f"But you requested up to {freq_end} Hz.\n\n"
                f"Frequencies above {max_achievable_freq:.2f} Hz will be skipped.\n\n"
                f"To test higher frequencies, reduce Stim Width and/or Read Delay.\n"
                f"Do you want to continue anyway?"
            )
            
            if not messagebox.askyesno("Frequency Range Warning", warning_msg):
                return
        
        # Collect base parameters - USE SRDP-SPECIFIC FIELDS
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different.")
        
        base_params = {
            "mode": "srdp",
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": validate_float(srdp_stim_level_entry.get(), "SRDP Stim Level"),
            "stim_width_ms": validate_float(srdp_stim_width_entry.get(), "SRDP Stim Width"),
            "n_pulses": int(validate_float(srdp_n_pulses_entry.get(), "SRDP # Pulses")),
            "read_voltage": validate_float(read_voltage_entry.get(), "Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5,
            "measure_avg": 1,       
            "wire_mode": wire_mode.get()  
        }
        
        # Run measurement or simulation
        if simulate_var.get():
            results = synapse_engine.simulate_srdp(base_params, freq_list.tolist())
            messagebox.showinfo("Simulation", "SRDP simulation completed!")
        else:
            port = port_entry.get().strip()
            connection = connection_type.get()
            
            if not port:
                raise ValueError("Port Address cannot be empty.")
            
            instrument = synapse_engine.open_instrument(port, connection)
            
            try:
                results = synapse_engine.measure_srdp(
                    instrument, stim_ch_str, read_ch_str, base_params, freq_list.tolist()
                )
                messagebox.showinfo("Success", "SRDP characterization completed!")
            finally:
                instrument.close()
        
        # Plot and store results
        plot_srdp_data(results)
        synapse_data_storage.append(results)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_stdp_characterization():
    """
    Runs STDP characterization with timing sweep.
    """
    global synapse_data_storage
    
    try:
        # Parse timing range
        dt_start = validate_float(stdp_dt_start_entry.get(), "STDP Start Δt")
        dt_end = validate_float(stdp_dt_end_entry.get(), "STDP End Δt")
        dt_points = int(validate_float(stdp_dt_points_entry.get(), "STDP Δt Points"))
        
        if dt_points < 2:
            raise ValueError("Need at least 2 timing points")
        
        # Generate Δt list
        delta_t_list = np.linspace(dt_start, dt_end, dt_points)
        
        # Collect base parameters - USE STDP-SPECIFIC FIELDS
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different.")
        
        base_params = {
            "mode": "stdp",
            "stim_drive_type": "V",  # STDP typically uses voltage
            "stim_level": validate_float(stdp_pre_level_entry.get(), "Pre-spike Level"),
            "post_spike_level": validate_float(stdp_post_level_entry.get(), "Post-spike Level"),
            "stim_width_ms": validate_float(stdp_pulse_width_entry.get(), "Pulse Width"),
            "stim_period_ms": validate_float(stdp_pair_period_entry.get(), "Pair Period"),
            "n_pulses": int(validate_float(stdp_n_pairs_entry.get(), "# Spike Pairs")),
            "read_voltage": validate_float(read_voltage_entry.get(), "Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5,
            "measure_avg": 1,
            "wire_mode": wire_mode.get()
        }
        
        # Run measurement or simulation
        if simulate_var.get():
            results = synapse_engine.simulate_stdp(base_params, delta_t_list.tolist())
            messagebox.showinfo("Simulation", "STDP simulation completed!")
        else:
            port = port_entry.get().strip()
            connection = connection_type.get()
            
            if not port:
                raise ValueError("Port Address cannot be empty.")
            
            instrument = synapse_engine.open_instrument(port, connection)
            
            try:
                results = synapse_engine.measure_stdp(
                    instrument, stim_ch_str, read_ch_str, base_params, delta_t_list.tolist()
                )
                messagebox.showinfo("Success", "STDP characterization completed!")
            finally:
                instrument.close()
        
        # Plot and store results
        plot_stdp_data(results)
        synapse_data_storage.append(results)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))



def run_cycle_characterization():
    """
    Runs potentiation-depression cycle characterization with real-time plotting.
    """
    global synapse_data_storage
    
    try:
        # Check if multi-device mode is enabled
        if multi_device_var.get():
            run_multi_device_characterization()
            return
        
        # Validate channels
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different.")
        
        # Get cycle parameters
        n_cycles = int(validate_float(cycle_n_cycles_entry.get(), "# Cycles"))
        inter_cycle_delay = validate_float(cycle_delay_entry.get(), "Inter-cycle Delay")
        
        if n_cycles <= 0:
            raise ValueError("Number of cycles must be greater than zero")
        
        # Collect potentiation parameters
        pot_params = {
            "mode": "electrical",
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": validate_float(pot_stim_level_entry.get(), "Pot Stim Level"),
            "stim_width_ms": validate_float(pot_stim_width_entry.get(), "Pot Stim Width"),
            "stim_period_ms": validate_float(pot_stim_period_entry.get(), "Pot Stim Period"),
            "n_pulses": int(validate_float(pot_n_pulses_entry.get(), "Pot # Pulses")),
            "read_voltage": validate_float(pot_read_voltage_entry.get(), "Pot Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5,
            "measure_avg": 1,
            "wire_mode": wire_mode.get()  
        }
        
        dep_params = {
            "mode": "electrical",
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": validate_float(dep_stim_level_entry.get(), "Dep Stim Level"),
            "stim_width_ms": validate_float(dep_stim_width_entry.get(), "Dep Stim Width"),
            "stim_period_ms": validate_float(dep_stim_period_entry.get(), "Dep Stim Period"),
            "n_pulses": int(validate_float(dep_n_pulses_entry.get(), "Dep # Pulses")),
            "read_voltage": validate_float(dep_read_voltage_entry.get(), "Dep Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5,
            "measure_avg": 1,
            "wire_mode": wire_mode.get() 
        }
        
        # Define real-time update callback
        def update_plot(results):
            plot_cycle_data(results)
            root.update()  # Force GUI update
        
        # Run measurement or simulation
        if simulate_var.get():
            results = synapse_cycle.simulate_potentiation_depression_cycle(
                pot_params, dep_params, n_cycles, update_callback=update_plot
            )
            messagebox.showinfo("Simulation", "Cycle simulation completed!")
        else:
            port = port_entry.get().strip()
            connection = connection_type.get()
            
            if not port:
                raise ValueError("Port Address cannot be empty.")
            
            instrument = synapse_engine.open_instrument(port, connection)
            
            try:
                results = synapse_cycle.run_potentiation_depression_cycle(
                    instrument, stim_ch_str, read_ch_str, 
                    pot_params, dep_params, n_cycles, inter_cycle_delay,
                    update_callback=update_plot
                )
                messagebox.showinfo("Success", "Cycle characterization completed!")
            finally:
                instrument.close()
        
        # Final plot update
        plot_cycle_data(results)
        
        # Store results
        synapse_data_storage.append(results)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_multi_device_characterization():
    """
    Runs cycling across multiple devices sequentially.
    """
    global synapse_data_storage
    
    try:
        # Get multi-device parameters
        n_devices = int(validate_float(n_devices_entry.get(), "# Devices"))
        inter_device_delay = validate_float(inter_device_delay_entry.get(), "Inter-device Delay")
        device_name_pattern = device_name_pattern_entry.get().strip()
        
        if n_devices <= 0:
            raise ValueError("Number of devices must be greater than zero")
        
        # Get cycle parameters (same for all devices)
        n_cycles = int(validate_float(cycle_n_cycles_entry.get(), "# Cycles"))
        inter_cycle_delay = validate_float(cycle_delay_entry.get(), "Inter-cycle Delay")
        
        # Get channels
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different.")
        
        # Collect parameters (same as single device)
        pot_params = {
            "mode": "electrical",
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": validate_float(pot_stim_level_entry.get(), "Pot Stim Level"),
            "stim_width_ms": validate_float(pot_stim_width_entry.get(), "Pot Stim Width"),
            "stim_period_ms": validate_float(pot_stim_period_entry.get(), "Pot Stim Period"),
            "n_pulses": int(validate_float(pot_n_pulses_entry.get(), "Pot # Pulses")),
            "read_voltage": validate_float(pot_read_voltage_entry.get(), "Pot Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5,
            "measure_avg": 1,
            "wire_mode": wire_mode.get()  # *** ADD THIS ***
        }
        
        dep_params = {
            "mode": "electrical",
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": validate_float(dep_stim_level_entry.get(), "Dep Stim Level"),
            "stim_width_ms": validate_float(dep_stim_width_entry.get(), "Dep Stim Width"),
            "stim_period_ms": validate_float(dep_stim_period_entry.get(), "Dep Stim Period"),
            "n_pulses": int(validate_float(dep_n_pulses_entry.get(), "Dep # Pulses")),
            "read_voltage": validate_float(dep_read_voltage_entry.get(), "Dep Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5,
            "measure_avg": 1,
            "wire_mode": wire_mode.get()  # *** ADD THIS ***
        }
        
        # Create device configurations
        device_configs = []
        for i in range(n_devices):
            config = {
                "device_name": f"{device_name_pattern}_{i+1}",
                "stim_ch": stim_ch_str,
                "read_ch": read_ch_str,
                "pot_params": pot_params.copy(),
                "dep_params": dep_params.copy(),
                "n_cycles": n_cycles,
                "inter_cycle_delay_ms": inter_cycle_delay,
                "inter_device_delay_ms": inter_device_delay
            }
            device_configs.append(config)
        
        # Define real-time update callback
        def update_plot(multi_results):
            plot_multi_device_data(multi_results)
            root.update()
        
        # Run measurement or simulation
        if simulate_var.get():
            results = synapse_cycle.simulate_multi_device_cycles(
                device_configs, update_callback=update_plot
            )
            messagebox.showinfo("Simulation", f"Multi-device simulation completed for {n_devices} devices!")
        else:
            port = port_entry.get().strip()
            connection = connection_type.get()
            
            if not port:
                raise ValueError("Port Address cannot be empty.")
            
            # Confirm with user before starting long measurement
            confirm = messagebox.askyesno(
                "Confirm Multi-Device Measurement",
                f"This will sequentially measure {n_devices} devices.\n"
                f"Each device will undergo {n_cycles} pot-dep cycles.\n\n"
                f"This may take a long time. Continue?"
            )
            
            if not confirm:
                return
            
            instrument = synapse_engine.open_instrument(port, connection)
            
            try:
                results = synapse_cycle.run_multi_device_cycles(
                    instrument, device_configs, update_callback=update_plot
                )
                messagebox.showinfo("Success", f"Multi-device characterization completed for {n_devices} devices!")
            finally:
                instrument.close()
        
        # Final plot update
        plot_multi_device_data(results)
        
        # Store results
        synapse_data_storage.append(results)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))





def run_transistor_measurement():

    global jv_curves
    try:
        gate_start = validate_float(gate_start_voltage.get(), "Gate Start Voltage")
        gate_end = validate_float(gate_end_voltage.get(), "Gate End Voltage")
        gate_step = validate_float(gate_voltage_step.get(), "Gate Voltage Step")
        channel = channel_selection.get()
        port = port_entry.get().strip()
        connection = connection_type.get()

        if not port:
            raise ValueError("Port Address cannot be empty.")

        rm = pyvisa.ResourceManager()
        instrument = None
        if connection == "GPIB":
            instrument = rm.open_resource(f"GPIB::{port}::INSTR")
        elif connection == "RS232":
            instrument = rm.open_resource(port, baud_rate=9600, data_bits=8, parity=pyvisa.constants.Parity.none, stop_bits=pyvisa.constants.StopBits.one, flow_control=pyvisa.constants.VI_ASRL_FLOW_NONE)
        elif connection == "LAN":
            instrument = rm.open_resource(f"TCPIP::{port}::INSTR")

        if instrument is None:
            raise ValueError("Could not establish communication with the instrument.")

        drain_channel = "smua" if channel == "Channel A" else "smub"
        gate_channel = "smub" if channel == "Channel A" else "smua"

        instrument.write("*CLS")

        instrument.write(f"{gate_channel}.source.func = {gate_channel}.OUTPUT_DCVOLTS")
        instrument.write(f"{gate_channel}.source.autorangev = {gate_channel}.AUTORANGE_ON")
        instrument.write(f"{gate_channel}.source.output = {gate_channel}.OUTPUT_ON")

        gate_voltages = np.arange(gate_start, gate_end + gate_step, gate_step)
        all_transistor_data = []

        for vgs in gate_voltages:
            vgs = round(vgs, 2)
            instrument.write(f"{gate_channel}.source.levelv = {vgs}")
            time.sleep(0.1)

            instrument.write(f"{gate_channel}.source.output = {gate_channel}.OUTPUT_ON")

            run_measurement_buffered(transistor_mode=True)

            if not jv_curves:
                print(f"⚠️ No JV data collected for V_GS = {vgs} V.")
                continue

            jv_curves[-1]["Gate Voltage (V)"] = vgs
            all_transistor_data.append(jv_curves[-1])

        instrument.write(f"{gate_channel}.source.output = {gate_channel}.OUTPUT_OFF")

        if not all_transistor_data:
            messagebox.showerror("Error", "No valid transistor JV data collected.")
            return

        jv_curves = all_transistor_data

        print(f"✅ Collected {len(jv_curves)} JV curves.")

        if not jv_curves:
            messagebox.showerror("Error", "No JV curves available for plotting.")
            return

        ax.clear()
        colors = plt.cm.viridis(np.linspace(0, 1, len(gate_voltages)))
        for i, entry in enumerate(jv_curves):
            if "Voltages" in entry and "Current Density" in entry:
                ax.plot(entry["Voltages"], entry["Current Density"], 
                        label=f"V_GS = {entry['Gate Voltage (V)']:.2f} V", color=colors[i])
            else:
                print(f"⚠️ Missing data for entry {i}")

        ax.set_title("Transistor JV Curves")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density (mA/cm²)")
        ax.legend(loc="upper left")
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        if 'instrument' in locals() and instrument is not None:
            instrument.close()

def save_transistor_data():

    try:
        file_name = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Transistor JV Data")
        if not file_name:
            return

        with open(file_name, "w") as f:
            if not jv_curves:
                messagebox.showerror("Save Error", "No JV curves available to save.")
                return

            try:
                area = validate_float(surface_area.get(), "Sample Surface Area")
            except ValueError:
                messagebox.showerror("Save Error", "Invalid surface area value.")
                return

            header_row = []
            for entry in jv_curves:
                if "Gate Voltage (V)" not in entry:
                    messagebox.showerror("Save Error", "'Gate Voltage (V)' key missing in data.")
                    return
                gate_voltage = entry["Gate Voltage (V)"]
                header_row.extend([f"V_GS = {gate_voltage} V", f"Surface Area = {area} cm²"])

            f.write(",".join(header_row) + "\n")

            column_labels = []
            for _ in jv_curves:
                column_labels.extend(["Voltage (V)", "Current Density (mA/cm²)"])
            f.write(",".join(column_labels) + "\n")

            max_length = max(len(entry["Voltages"]) for entry in jv_curves)

            for i in range(max_length):
                row = []
                for entry in jv_curves:
                    voltages = entry["Voltages"]
                    currents = entry["Current Density"]
                    row.append(str(voltages[i]) if i < len(voltages) else "")
                    row.append(str(currents[i]) if i < len(currents) else "")
                f.write(",".join(row) + "\n")

        messagebox.showinfo("Save Successful", "Transistor JV data saved successfully.")

    except Exception as e:
        messagebox.showerror("Save Error", str(e))

def run_synapse_mode():
    """
    Runs synapse measurement using the synapse_engine module.
    Handles both hardware measurements and simulation.
    """
    global synapse_data_storage
    
    try:
        # Validate channel selection
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different (smua/smub).")
        
        # Parse and validate all parameters
        stim_level = validate_float(stim_level_entry.get(), "Stim Level")
        stim_width = validate_float(stim_width_entry.get(), "Stim Width")
        stim_period = validate_float(stim_period_entry.get(), "Stim Period")
        n_pulses = int(validate_float(n_pulses_entry.get(), "# Pulses"))
        read_voltage = validate_float(read_voltage_entry.get(), "Read Voltage")
        read_delay = validate_float(read_delay_entry.get(), "Read Delay")
        compliance_A = validate_float(compliance_synapse_entry.get(), "Compliance")
        
        # Validation checks
        if stim_period < stim_width:
            raise ValueError("Stim Period must be >= Stim Width")
        
        if n_pulses <= 0:
            raise ValueError("Number of pulses must be greater than zero")
        
        if compliance_A <= 0:
            raise ValueError("Compliance must be greater than zero")
        
        # Safety check for high stimulus levels
        if not synapse_engine.safety_check(stim_level, max_safe_voltage=2.5):
            return
        
        # Collect parameters
        params = {
            "mode": synapse_mode_selection.get().lower(),
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": stim_level,
            "stim_width_ms": stim_width,
            "stim_period_ms": stim_period,
            "n_pulses": n_pulses,
            "read_voltage": read_voltage,
            "read_delay_ms": read_delay,
            "compliance_A": compliance_A,
            "measure_avg": 1,
            "settle_ms": 5,
            "wire_mode": wire_mode.get() 
        }
                
        # Run measurement or simulation
        if simulate_var.get():
            # Simulation mode
            results = synapse_engine.simulate_pulse_read(params)
            messagebox.showinfo("Simulation", "Simulation completed successfully!")
        else:
            # Hardware mode
            port = port_entry.get().strip()
            connection = connection_type.get()
            
            if not port:
                raise ValueError("Port Address cannot be empty.")
            
            # Open instrument
            instrument = synapse_engine.open_instrument(port, connection)
            
            # Configure timeout
            try:
                user_timeout = validate_float(timeout_entry.get(), "Timeout Duration")
                if user_timeout <= 0:
                    raise ValueError("Timeout duration must be greater than zero.")
                timeout_value = user_timeout * 1000
            except ValueError:
                total_time = (params['stim_period_ms'] * params['n_pulses']) / 1000.0
                timeout_value = max(5000, total_time * 1000 * 1.5)
            
            instrument.timeout = timeout_value
            
            try:
                # Run the pulse-read sequence
                results = synapse_engine.pulse_read_sequence(
                    instrument, 
                    stim_ch_str, 
                    read_ch_str, 
                    params
                )
                messagebox.showinfo("Success", "Synapse measurement completed successfully!")
                
            finally:
                # Close instrument
                instrument.close()
        
        # Plot results
        plot_synapse_data(results)
        
        # Store results for saving
        synapse_data_storage.append(results)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_synapse_data():
    """
    Saves synapse measurement data to CSV file with metadata and metrics.
    Handles regular synapse, SRDP, and STDP data.
    """
    global synapse_data_storage
    
    try:
        if not synapse_data_storage:
            messagebox.showerror("Save Error", "No synapse data available to save.")
            return
        
        file_name = asksaveasfilename(
            defaultextension=".csv", 
            filetypes=[("CSV files", "*.csv")], 
            title="Save Synapse/Memristor Data"
        )
        
        if not file_name:
            return
        
        # Get the most recent measurement
        last_data = synapse_data_storage[-1]
        
        # Determine data type and save accordingly
        if "frequencies_hz" in last_data:
            # SRDP data
            save_srdp_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful", 
                f"SRDP data saved successfully to:\n{file_name}"
            )
        elif "delta_t_ms" in last_data:
            # STDP data
            save_stdp_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful", 
                f"STDP data saved successfully to:\n{file_name}"
            )
        elif "cycles" in last_data and "devices" not in last_data:
            # Single-device cycle data
            synapse_cycle.save_cycle_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful", 
                f"Cycle data saved successfully to:\n{file_name}"
            )
        elif "devices" in last_data:
            # Multi-device cycle data
            synapse_cycle.save_multi_device_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful", 
                f"Multi-device data saved successfully to:\n{file_name}"
            )
        else:
            # Regular synapse pulse-read data
            metrics = synapse_engine.process_and_save_synapse_data(last_data, file_name)
            messagebox.showinfo(
                "Save Successful", 
                f"Synapse data saved successfully to:\n{file_name}\n\n"
                f"Metrics calculated:\n" + 
                "\n".join(f"  • {k}: {v}" for k, v in list(metrics.items())[:3])
            )
        
        # Clear stored data
        synapse_data_storage.clear()
        
    except Exception as e:
        messagebox.showerror("Save Error", str(e))


def save_srdp_data(results, filename):
    """
    Saves SRDP characterization data to CSV.
    
    Args:
        results (dict): SRDP results from measure_srdp or simulate_srdp
        filename (str): Path to save CSV file
    """
    csv_content = []
    
    # Metadata header
    params = results["base_params"]
    metadata_line = "# SRDP Characterization # " + " # ".join(f"{k}={v}" for k, v in params.items())
    csv_content.append(metadata_line)
    
    # Column headers
    csv_content.append("frequency_Hz,delta_G_S,delta_G_percent,G_initial_S,G_final_S")
    
    # Data rows
    for i in range(len(results["frequencies_hz"])):
        row = [
            f"{results['frequencies_hz'][i]:.4f}",
            f"{results['delta_g_S'][i]:.6e}",
            f"{results['delta_g_percent'][i]:.4f}",
            f"{results['g_initial_S'][i]:.6e}",
            f"{results['g_final_S'][i]:.6e}"
        ]
        csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(csv_content))


def save_stdp_data(results, filename):
    """
    Saves STDP characterization data to CSV.
    
    Args:
        results (dict): STDP results from measure_stdp or simulate_stdp
        filename (str): Path to save CSV file
    """
    csv_content = []
    
    # Metadata header
    params = results["base_params"]
    metadata_line = "# STDP Characterization # " + " # ".join(f"{k}={v}" for k, v in params.items())
    csv_content.append(metadata_line)
    
    # Column headers
    csv_content.append("delta_t_ms,delta_G_S,delta_G_percent,G_initial_S,G_final_S")
    
    # Data rows
    for i in range(len(results["delta_t_ms"])):
        row = [
            f"{results['delta_t_ms'][i]:.4f}",
            f"{results['delta_g_S'][i]:.6e}",
            f"{results['delta_g_percent'][i]:.4f}",
            f"{results['g_initial_S'][i]:.6e}",
            f"{results['g_final_S'][i]:.6e}"
        ]
        csv_content.append(",".join(row))
    
    # Write to file
    with open(filename, "w") as f:
        f.write("\n".join(csv_content))


def toggle_mode(event=None):
    if mode_selection.get() == "Transistor":
        gate_start_voltage.grid()
        gate_end_voltage.grid()
        gate_voltage_step.grid()
    else:
        gate_start_voltage.grid_remove()
        gate_end_voltage.grid_remove()
        gate_voltage_step.grid_remove()
        
def toggle_multi_device_mode():
    """Enables/disables multi-device controls."""
    if multi_device_var.get():
        n_devices_entry.configure(state="normal")
        inter_device_delay_entry.configure(state="normal")
        device_name_pattern_entry.configure(state="normal")
    else:
        n_devices_entry.configure(state="disabled")
        inter_device_delay_entry.configure(state="disabled")
        device_name_pattern_entry.configure(state="disabled")


def apply_preset(preset_name):
    """Applies predefined parameter presets for common synapse measurements."""
    if preset_name == "LTP Moderate":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "1.0")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "100")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "200")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "50")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "10")
        
    elif preset_name == "LTD Moderate":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "-1.0")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "100")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "200")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "50")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "10")
        
    elif preset_name == "PPF Test":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "1.5")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "10")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "30")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "2")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "5")
        
    elif preset_name == "High Speed":
        stim_level_entry.delete(0, 'end')
        stim_level_entry.insert(0, "1.0")
        stim_width_entry.delete(0, 'end')
        stim_width_entry.insert(0, "10")
        stim_period_entry.delete(0, 'end')
        stim_period_entry.insert(0, "20")
        n_pulses_entry.delete(0, 'end')
        n_pulses_entry.insert(0, "100")
        read_voltage_entry.delete(0, 'end')
        read_voltage_entry.insert(0, "0.1")
        read_delay_entry.delete(0, 'end')
        read_delay_entry.insert(0, "2")


def apply_cycle_preset(preset_name):
    """Applies cycle parameter presets."""
    if preset_name == "Custom":
        return
    
    try:
        pot_params, dep_params, n_cycles, inter_delay = synapse_cycle.get_preset_parameters(preset_name)
        
        # Update potentiation fields
        pot_stim_level_entry.delete(0, 'end')
        pot_stim_level_entry.insert(0, str(pot_params['stim_level']))
        
        pot_stim_width_entry.delete(0, 'end')
        pot_stim_width_entry.insert(0, str(pot_params['stim_width_ms']))
        
        pot_stim_period_entry.delete(0, 'end')
        pot_stim_period_entry.insert(0, str(pot_params['stim_period_ms']))
        
        pot_n_pulses_entry.delete(0, 'end')
        pot_n_pulses_entry.insert(0, str(pot_params['n_pulses']))
        
        pot_read_voltage_entry.delete(0, 'end')
        pot_read_voltage_entry.insert(0, str(pot_params['read_voltage']))
        
        # Update depression fields
        dep_stim_level_entry.delete(0, 'end')
        dep_stim_level_entry.insert(0, str(dep_params['stim_level']))
        
        dep_stim_width_entry.delete(0, 'end')
        dep_stim_width_entry.insert(0, str(dep_params['stim_width_ms']))
        
        dep_stim_period_entry.delete(0, 'end')
        dep_stim_period_entry.insert(0, str(dep_params['stim_period_ms']))
        
        dep_n_pulses_entry.delete(0, 'end')
        dep_n_pulses_entry.insert(0, str(dep_params['n_pulses']))
        
        dep_read_voltage_entry.delete(0, 'end')
        dep_read_voltage_entry.insert(0, str(dep_params['read_voltage']))
        
        # Update cycle parameters
        cycle_n_cycles_entry.delete(0, 'end')
        cycle_n_cycles_entry.insert(0, str(n_cycles))
        
        cycle_delay_entry.delete(0, 'end')
        cycle_delay_entry.insert(0, str(inter_delay))
        
        messagebox.showinfo("Preset Applied", f"'{preset_name}' preset loaded successfully!")
        
    except Exception as e:
        messagebox.showerror("Preset Error", f"Failed to apply preset: {str(e)}")


# ==============================================================================
# GUI SETUP
# ==============================================================================

# Create main application window
root = ctk.CTk()
root.title("Shockingly Accurate IV")
root.geometry("1400x700")

# Main container with two columns
main_container = ctk.CTkFrame(root)
main_container.pack(fill="both", expand=True, padx=10, pady=10)

# Left panel for inputs
left_panel = ctk.CTkScrollableFrame(main_container, width=350)
left_panel.pack(side="left", fill="both", expand=False, padx=(0, 10))

# Right panel for graphs and parameters
right_panel = ctk.CTkFrame(main_container)
right_panel.pack(side="left", fill="both", expand=True)

# Create horizontal split: graph on left, sidebar on right
graph_and_results_container = ctk.CTkFrame(right_panel)
graph_and_results_container.pack(fill="both", expand=True)

# Right sidebar for results and logo
right_sidebar = ctk.CTkFrame(graph_and_results_container, width=300)
right_sidebar.pack(side="right", fill="both", padx=(10, 0))
right_sidebar.pack_propagate(False)  # Maintain fixed width

# Graph frame (top of right panel)
graph_frame = ctk.CTkFrame(graph_and_results_container)
graph_frame.pack(side="left", fill="both", expand=True)

# Parameters display frame (top of right sidebar)
params_frame = ctk.CTkFrame(right_sidebar)
params_frame.pack(fill="both", expand=True, pady=(0, 10))
params_text = ctk.CTkTextbox(params_frame, height=150)
params_text.pack(fill="both", expand=True, padx=5, pady=5)

# Logo frame (bottom of right sidebar)
logo_frame = ctk.CTkFrame(right_sidebar, height=150)
logo_frame.pack(fill="x", expand=False)
logo_frame.pack_propagate(False)  # Maintain fixed height

try:
    from PIL import Image
    logo_image = ctk.CTkImage(light_image=Image.open("logo.png"),
                              dark_image=Image.open("logo.png"),
                              size=(180, 130))
    logo_label = ctk.CTkLabel(logo_frame, image=logo_image, text="")
    logo_label.pack(expand=True, pady=10)
except Exception as e:
    # Fallback if logo not found
    logo_label = ctk.CTkLabel(logo_frame, text="Logo", font=("Arial", 20))
    logo_label.pack(expand=True)



# ==============================================================================
# COMMON PARAMETERS (Always Visible)
# ==============================================================================

ctk.CTkLabel(left_panel, text="GENERAL PARAMETERS", font=("Arial", 14, "bold")).pack(pady=(5, 10))

# Connection Type
ctk.CTkLabel(left_panel, text="Connection Type:").pack(anchor="w", padx=10)
connection_type = ctk.CTkComboBox(left_panel, values=["GPIB", "RS232", "LAN"])
connection_type.pack(fill="x", padx=10, pady=5)
connection_type.set("GPIB")

# Port Address
ctk.CTkLabel(left_panel, text="Port Address:").pack(anchor="w", padx=10)
port_entry = ctk.CTkEntry(left_panel)
port_entry.pack(fill="x", padx=10, pady=5)

# Channel Selection
ctk.CTkLabel(left_panel, text="Channel:").pack(anchor="w", padx=10)
channel_selection = ctk.CTkComboBox(left_panel, values=["Channel A", "Channel B"])
channel_selection.pack(fill="x", padx=10, pady=5)
channel_selection.set("Channel A")

# Add clarification label for transistor mode
channel_info_label = ctk.CTkLabel(
    left_panel, 
    text="ℹ️ Selected channel = Drain, Other channel = Gate",
    font=("Arial", 12, "italic"),
    text_color="gray"
)
channel_info_label.pack(anchor="w", padx=10, pady=(0, 5))

# NPLC
ctk.CTkLabel(left_panel, text="Measurement Speed (NPLC):").pack(anchor="w", padx=10)
nplc_entry = ctk.CTkEntry(left_panel)
nplc_entry.pack(fill="x", padx=10, pady=5)

# Timeout
ctk.CTkLabel(left_panel, text="Timeout Duration (s):").pack(anchor="w", padx=10)
timeout_entry = ctk.CTkEntry(left_panel)
timeout_entry.pack(fill="x", padx=10, pady=5)
timeout_entry.insert(0, "30")

# Compliance
ctk.CTkLabel(left_panel, text="Compliance (A):").pack(anchor="w", padx=10)
compliance = ctk.CTkComboBox(left_panel, values=["100nA", "1µA", "10µA", "100µA", "1mA", "10mA", "100mA", "1A", "1.5A"])
compliance.pack(fill="x", padx=10, pady=5)
compliance.set("1mA")

# Cell #
ctk.CTkLabel(left_panel, text="Cell #:").pack(anchor="w", padx=10)
sample_name_entry = ctk.CTkEntry(left_panel)
sample_name_entry.pack(fill="x", padx=10, pady=5)

# Sample Surface Area
ctk.CTkLabel(left_panel, text="Cell Surface Area (cm²):").pack(anchor="w", padx=10)
surface_area = ctk.CTkEntry(left_panel)
surface_area.pack(fill="x", padx=10, pady=5)

# Wire mode
wire_mode = ctk.CTkComboBox(left_panel, values=["2-Wire", "4-Wire"])
wire_mode.pack(fill="x", padx=10, pady=5)
wire_mode.set("4-Wire")

# Autorange
autorange_var = ctk.IntVar(value=0)
autorange_check = ctk.CTkCheckBox(left_panel, text="Enable Autorange", variable=autorange_var)
autorange_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(left_panel, text="─" * 50).pack(pady=10)

# ==============================================================================
# MODE SELECTION
# ==============================================================================

ctk.CTkLabel(left_panel, text="MEASUREMENT MODE", font=("Arial", 14, "bold")).pack(pady=(5, 10))

mode_selection = ctk.CTkComboBox(left_panel, values=["Diode", "Transistor", "Synapse"], command=lambda x: update_mode_display())
mode_selection.pack(fill="x", padx=10, pady=5)
mode_selection.set("Diode")

# ==============================================================================
# DIODE MODE PARAMETERS
# ==============================================================================

diode_frame = ctk.CTkFrame(left_panel)

ctk.CTkLabel(diode_frame, text="DIODE PARAMETERS", font=("Arial", 12, "bold")).pack(pady=(5, 10))

ctk.CTkLabel(diode_frame, text="Starting Voltage (V):").pack(anchor="w", padx=10)
start_voltage = ctk.CTkEntry(diode_frame)
start_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Ending Voltage (V):").pack(anchor="w", padx=10)
end_voltage = ctk.CTkEntry(diode_frame)
end_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Voltage Step (V):").pack(anchor="w", padx=10)
voltage_step = ctk.CTkEntry(diode_frame)
voltage_step.pack(fill="x", padx=10, pady=5)

hysteresis_var = ctk.IntVar()
hysteresis_check = ctk.CTkCheckBox(diode_frame, text="Perform Hysteresis", variable=hysteresis_var)
hysteresis_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Hysteresis Cycles:").pack(anchor="w", padx=10)
hysteresis_cycles = ctk.CTkEntry(diode_frame)
hysteresis_cycles.pack(fill="x", padx=10, pady=5)

dark_measurement = ctk.IntVar()
dark_check = ctk.CTkCheckBox(diode_frame, text="Dark JV", variable=dark_measurement)
dark_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(diode_frame, text="Irradiance (W/m²):").pack(anchor="w", padx=10)
light_power_entry = ctk.CTkEntry(diode_frame)
light_power_entry.pack(fill="x", padx=10, pady=5)
light_power_entry.insert(0, "1000")

# Diode buttons
diode_button_frame = ctk.CTkFrame(diode_frame)
diode_button_frame.pack(fill="x", padx=10, pady=10)

run_diode_button = ctk.CTkButton(diode_button_frame, text="Run Measurement", command=run_measurement_buffered)
run_diode_button.pack(fill="x", pady=5)

save_diode_button = ctk.CTkButton(diode_button_frame, text="Save Data", command=save_curve)
save_diode_button.pack(fill="x", pady=5)

# ==============================================================================
# TRANSISTOR MODE PARAMETERS
# ==============================================================================

transistor_frame = ctk.CTkFrame(left_panel)

ctk.CTkLabel(transistor_frame, text="TRANSISTOR PARAMETERS", font=("Arial", 12, "bold")).pack(pady=(5, 10))

ctk.CTkLabel(transistor_frame, text="Starting Voltage (V):").pack(anchor="w", padx=10)
start_voltage_trans = ctk.CTkEntry(transistor_frame)
start_voltage_trans.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Ending Voltage (V):").pack(anchor="w", padx=10)
end_voltage_trans = ctk.CTkEntry(transistor_frame)
end_voltage_trans.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Voltage Step (V):").pack(anchor="w", padx=10)
voltage_step_trans = ctk.CTkEntry(transistor_frame)
voltage_step_trans.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Gate Start Voltage (V):").pack(anchor="w", padx=10)
gate_start_voltage = ctk.CTkEntry(transistor_frame)
gate_start_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Gate End Voltage (V):").pack(anchor="w", padx=10)
gate_end_voltage = ctk.CTkEntry(transistor_frame)
gate_end_voltage.pack(fill="x", padx=10, pady=5)

ctk.CTkLabel(transistor_frame, text="Gate Voltage Step (V):").pack(anchor="w", padx=10)
gate_voltage_step = ctk.CTkEntry(transistor_frame)
gate_voltage_step.pack(fill="x", padx=10, pady=5)

# Transistor buttons
transistor_button_frame = ctk.CTkFrame(transistor_frame)
transistor_button_frame.pack(fill="x", padx=10, pady=10)

run_transistor_button = ctk.CTkButton(transistor_button_frame, text="Run Measurement", command=run_transistor_measurement)
run_transistor_button.pack(fill="x", pady=5)

save_transistor_button = ctk.CTkButton(transistor_button_frame, text="Save Data", command=save_transistor_data)
save_transistor_button.pack(fill="x", pady=5)

# ==============================================================================
# SYNAPSE MODE PARAMETERS
# ==============================================================================

synapse_frame = ctk.CTkFrame(left_panel)

ctk.CTkLabel(synapse_frame, text="SYNAPSE MODE", font=("Arial", 12, "bold")).pack(pady=(5, 10))

ctk.CTkLabel(synapse_frame, text="Synapse Sub-mode:").pack(anchor="w", padx=10)
synapse_submode = ctk.CTkComboBox(synapse_frame, values=["Basic", "SRDP", "STDP", "Cycle", "Multi-Device"], 
                                   command=lambda x: update_synapse_submode())
synapse_submode.pack(fill="x", padx=10, pady=5)
synapse_submode.set("Basic")

# Common synapse parameters
ctk.CTkLabel(synapse_frame, text="─ Common Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_frame, text="Synapse Type:").pack(anchor="w", padx=10)
synapse_mode_selection = ctk.CTkComboBox(synapse_frame, values=["Electrical", "Visual", "Memristor (Pulse)"])
synapse_mode_selection.pack(fill="x", padx=10, pady=5)
synapse_mode_selection.set("Electrical")

ctk.CTkLabel(synapse_frame, text="Stim Channel:").pack(anchor="w", padx=10)
stim_channel_var = ctk.CTkComboBox(synapse_frame, values=["smua", "smub"])
stim_channel_var.pack(fill="x", padx=10, pady=5)
stim_channel_var.set("smua")

ctk.CTkLabel(synapse_frame, text="Read Channel:").pack(anchor="w", padx=10)
read_channel_var = ctk.CTkComboBox(synapse_frame, values=["smua", "smub"])
read_channel_var.pack(fill="x", padx=10, pady=5)
read_channel_var.set("smub")

ctk.CTkLabel(synapse_frame, text="Stim Drive Type:").pack(anchor="w", padx=10)
stim_drive_var = ctk.CTkComboBox(synapse_frame, values=["V", "I"])
stim_drive_var.pack(fill="x", padx=10, pady=5)
stim_drive_var.set("V")

ctk.CTkLabel(synapse_frame, text="Read Voltage (V):").pack(anchor="w", padx=10)
read_voltage_entry = ctk.CTkEntry(synapse_frame)
read_voltage_entry.pack(fill="x", padx=10, pady=5)
read_voltage_entry.insert(0, "0.1")

ctk.CTkLabel(synapse_frame, text="Read Delay (ms):").pack(anchor="w", padx=10)
read_delay_entry = ctk.CTkEntry(synapse_frame)
read_delay_entry.pack(fill="x", padx=10, pady=5)
read_delay_entry.insert(0, "10")

ctk.CTkLabel(synapse_frame, text="Compliance (A):").pack(anchor="w", padx=10)
compliance_synapse_entry = ctk.CTkEntry(synapse_frame)
compliance_synapse_entry.pack(fill="x", padx=10, pady=5)
compliance_synapse_entry.insert(0, "100e-6")

simulate_var = ctk.IntVar()
simulate_check = ctk.CTkCheckBox(synapse_frame, text="Simulate (No Hardware)", variable=simulate_var)
simulate_check.pack(anchor="w", padx=10, pady=5)

# --- BASIC SYNAPSE SUBMODE ---
synapse_basic_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_basic_frame, text="─ Basic Synapse ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

preset_frame = ctk.CTkFrame(synapse_basic_frame)
preset_frame.pack(fill="x", padx=10, pady=5)
ctk.CTkLabel(preset_frame, text="Presets:").pack(anchor="w")
preset_var = ctk.CTkComboBox(preset_frame, values=["Custom", "LTP Moderate", "LTD Moderate", "PPF Test", "High Speed"],
                              command=lambda choice: apply_preset(choice))
preset_var.pack(fill="x")
preset_var.set("Custom")

ctk.CTkLabel(synapse_basic_frame, text="Stim Level (V or A):").pack(anchor="w", padx=10)
stim_level_entry = ctk.CTkEntry(synapse_basic_frame)
stim_level_entry.pack(fill="x", padx=10, pady=5)
stim_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_basic_frame, text="Stim Width (ms):").pack(anchor="w", padx=10)
stim_width_entry = ctk.CTkEntry(synapse_basic_frame)
stim_width_entry.pack(fill="x", padx=10, pady=5)
stim_width_entry.insert(0, "100")

ctk.CTkLabel(synapse_basic_frame, text="Stim Period (ms):").pack(anchor="w", padx=10)
stim_period_entry = ctk.CTkEntry(synapse_basic_frame)
stim_period_entry.pack(fill="x", padx=10, pady=5)
stim_period_entry.insert(0, "200")

ctk.CTkLabel(synapse_basic_frame, text="# Pulses:").pack(anchor="w", padx=10)
n_pulses_entry = ctk.CTkEntry(synapse_basic_frame)
n_pulses_entry.pack(fill="x", padx=10, pady=5)
n_pulses_entry.insert(0, "50")

synapse_basic_button_frame = ctk.CTkFrame(synapse_basic_frame)
synapse_basic_button_frame.pack(fill="x", padx=10, pady=10)

run_synapse_button = ctk.CTkButton(synapse_basic_button_frame, text="Run Measurement", command=run_synapse_mode, fg_color="#2E7D32", hover_color="#1B5E20")
run_synapse_button.pack(fill="x", pady=5)

save_synapse_button = ctk.CTkButton(synapse_basic_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_synapse_button.pack(fill="x", pady=5)

# --- SRDP SUBMODE ---
synapse_srdp_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_srdp_frame, text="─ SRDP Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_srdp_frame, text="Freq Start (Hz):").pack(anchor="w", padx=10)
srdp_freq_start_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_freq_start_entry.pack(fill="x", padx=10, pady=5)
srdp_freq_start_entry.insert(0, "1")

ctk.CTkLabel(synapse_srdp_frame, text="Freq End (Hz):").pack(anchor="w", padx=10)
srdp_freq_end_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_freq_end_entry.pack(fill="x", padx=10, pady=5)
srdp_freq_end_entry.insert(0, "100")

ctk.CTkLabel(synapse_srdp_frame, text="# Freq Points:").pack(anchor="w", padx=10)
srdp_freq_points_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_freq_points_entry.pack(fill="x", padx=10, pady=5)
srdp_freq_points_entry.insert(0, "10")

srdp_log_scale_var = ctk.IntVar(value=1)
srdp_log_scale_check = ctk.CTkCheckBox(synapse_srdp_frame, text="Log Scale Frequency", variable=srdp_log_scale_var)
srdp_log_scale_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(synapse_srdp_frame, text="Stim Level (V):").pack(anchor="w", padx=10)
srdp_stim_level_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_stim_level_entry.pack(fill="x", padx=10, pady=5)
srdp_stim_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_srdp_frame, text="Stim Width (ms):").pack(anchor="w", padx=10)
srdp_stim_width_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_stim_width_entry.pack(fill="x", padx=10, pady=5)
srdp_stim_width_entry.insert(0, "10")  # *** CHANGED FROM 100 TO 10 ***

ctk.CTkLabel(synapse_srdp_frame, text="# Pulses per freq:").pack(anchor="w", padx=10)
srdp_n_pulses_entry = ctk.CTkEntry(synapse_srdp_frame)
srdp_n_pulses_entry.pack(fill="x", padx=10, pady=5)
srdp_n_pulses_entry.insert(0, "50")

synapse_srdp_button_frame = ctk.CTkFrame(synapse_srdp_frame)
synapse_srdp_button_frame.pack(fill="x", padx=10, pady=10)

run_srdp_button = ctk.CTkButton(synapse_srdp_button_frame, text="Run SRDP", command=run_srdp_characterization, fg_color="#FF6F00", hover_color="#E65100")
run_srdp_button.pack(fill="x", pady=5)

save_srdp_button = ctk.CTkButton(synapse_srdp_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_srdp_button.pack(fill="x", pady=5)

# --- STDP SUBMODE ---
synapse_stdp_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_stdp_frame, text="─ STDP Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

ctk.CTkLabel(synapse_stdp_frame, text="Δt Start (ms):").pack(anchor="w", padx=10)
stdp_dt_start_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_dt_start_entry.pack(fill="x", padx=10, pady=5)
stdp_dt_start_entry.insert(0, "-50")

ctk.CTkLabel(synapse_stdp_frame, text="Δt End (ms):").pack(anchor="w", padx=10)
stdp_dt_end_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_dt_end_entry.pack(fill="x", padx=10, pady=5)
stdp_dt_end_entry.insert(0, "50")

ctk.CTkLabel(synapse_stdp_frame, text="# Δt Points:").pack(anchor="w", padx=10)
stdp_dt_points_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_dt_points_entry.pack(fill="x", padx=10, pady=5)
stdp_dt_points_entry.insert(0, "15")

ctk.CTkLabel(synapse_stdp_frame, text="# Spike Pairs:").pack(anchor="w", padx=10)
stdp_n_pairs_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_n_pairs_entry.pack(fill="x", padx=10, pady=5)
stdp_n_pairs_entry.insert(0, "50")

ctk.CTkLabel(synapse_stdp_frame, text="Pre-spike Level (V):").pack(anchor="w", padx=10)
stdp_pre_level_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_pre_level_entry.pack(fill="x", padx=10, pady=5)
stdp_pre_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_stdp_frame, text="Post-spike Level (V):").pack(anchor="w", padx=10)
stdp_post_level_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_post_level_entry.pack(fill="x", padx=10, pady=5)
stdp_post_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_stdp_frame, text="Pulse Width (ms):").pack(anchor="w", padx=10)
stdp_pulse_width_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_pulse_width_entry.pack(fill="x", padx=10, pady=5)
stdp_pulse_width_entry.insert(0, "10")

ctk.CTkLabel(synapse_stdp_frame, text="Pair Period (ms):").pack(anchor="w", padx=10)
stdp_pair_period_entry = ctk.CTkEntry(synapse_stdp_frame)
stdp_pair_period_entry.pack(fill="x", padx=10, pady=5)
stdp_pair_period_entry.insert(0, "100")

synapse_stdp_button_frame = ctk.CTkFrame(synapse_stdp_frame)
synapse_stdp_button_frame.pack(fill="x", padx=10, pady=10)

run_stdp_button = ctk.CTkButton(synapse_stdp_button_frame, text="Run STDP", command=run_stdp_characterization, fg_color="#7B1FA2", hover_color="#4A148C")
run_stdp_button.pack(fill="x", pady=5)

save_stdp_button = ctk.CTkButton(synapse_stdp_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_stdp_button.pack(fill="x", pady=5)

# --- CYCLE SUBMODE ---
synapse_cycle_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_cycle_frame, text="─ Cycle Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

cycle_preset_frame = ctk.CTkFrame(synapse_cycle_frame)
cycle_preset_frame.pack(fill="x", padx=10, pady=5)
ctk.CTkLabel(cycle_preset_frame, text="Cycle Presets:").pack(anchor="w")
cycle_preset_var = ctk.CTkComboBox(cycle_preset_frame, values=["Custom", "Standard Cycle", "Fast Cycle", "High Endurance", "Asymmetric"],
                                    command=lambda choice: apply_cycle_preset(choice))
cycle_preset_var.pack(fill="x")
cycle_preset_var.set("Custom")

ctk.CTkLabel(synapse_cycle_frame, text="# Pot-Dep Cycles:").pack(anchor="w", padx=10)
cycle_n_cycles_entry = ctk.CTkEntry(synapse_cycle_frame)
cycle_n_cycles_entry.pack(fill="x", padx=10, pady=5)
cycle_n_cycles_entry.insert(0, "3")

ctk.CTkLabel(synapse_cycle_frame, text="Inter-cycle Delay (ms):").pack(anchor="w", padx=10)
cycle_delay_entry = ctk.CTkEntry(synapse_cycle_frame)
cycle_delay_entry.pack(fill="x", padx=10, pady=5)
cycle_delay_entry.insert(0, "1000")

ctk.CTkLabel(synapse_cycle_frame, text="POTENTIATION", font=("Arial", 10, "bold")).pack(pady=(10, 2))

ctk.CTkLabel(synapse_cycle_frame, text="Pot Stim Level (V):").pack(anchor="w", padx=10)
pot_stim_level_entry = ctk.CTkEntry(synapse_cycle_frame)
pot_stim_level_entry.pack(fill="x", padx=10, pady=5)
pot_stim_level_entry.insert(0, "1.0")

ctk.CTkLabel(synapse_cycle_frame, text="Pot Stim Width (ms):").pack(anchor="w", padx=10)
pot_stim_width_entry = ctk.CTkEntry(synapse_cycle_frame)
pot_stim_width_entry.pack(fill="x", padx=10, pady=5)
pot_stim_width_entry.insert(0, "100")

ctk.CTkLabel(synapse_cycle_frame, text="Pot Stim Period (ms):").pack(anchor="w", padx=10)
pot_stim_period_entry = ctk.CTkEntry(synapse_cycle_frame)
pot_stim_period_entry.pack(fill="x", padx=10, pady=5)
pot_stim_period_entry.insert(0, "200")

ctk.CTkLabel(synapse_cycle_frame, text="Pot # Pulses:").pack(anchor="w", padx=10)
pot_n_pulses_entry = ctk.CTkEntry(synapse_cycle_frame)
pot_n_pulses_entry.pack(fill="x", padx=10, pady=5)
pot_n_pulses_entry.insert(0, "50")

ctk.CTkLabel(synapse_cycle_frame, text="Pot Read Voltage (V):").pack(anchor="w", padx=10)
pot_read_voltage_entry = ctk.CTkEntry(synapse_cycle_frame)
pot_read_voltage_entry.pack(fill="x", padx=10, pady=5)
pot_read_voltage_entry.insert(0, "0.1")

ctk.CTkLabel(synapse_cycle_frame, text="DEPRESSION", font=("Arial", 10, "bold")).pack(pady=(10, 2))

ctk.CTkLabel(synapse_cycle_frame, text="Dep Stim Level (V):").pack(anchor="w", padx=10)
dep_stim_level_entry = ctk.CTkEntry(synapse_cycle_frame)
dep_stim_level_entry.pack(fill="x", padx=10, pady=5)
dep_stim_level_entry.insert(0, "-1.0")

ctk.CTkLabel(synapse_cycle_frame, text="Dep Stim Width (ms):").pack(anchor="w", padx=10)
dep_stim_width_entry = ctk.CTkEntry(synapse_cycle_frame)
dep_stim_width_entry.pack(fill="x", padx=10, pady=5)
dep_stim_width_entry.insert(0, "100")

ctk.CTkLabel(synapse_cycle_frame, text="Dep Stim Period (ms):").pack(anchor="w", padx=10)
dep_stim_period_entry = ctk.CTkEntry(synapse_cycle_frame)
dep_stim_period_entry.pack(fill="x", padx=10, pady=5)
dep_stim_period_entry.insert(0, "200")

ctk.CTkLabel(synapse_cycle_frame, text="Dep # Pulses:").pack(anchor="w", padx=10)
dep_n_pulses_entry = ctk.CTkEntry(synapse_cycle_frame)
dep_n_pulses_entry.pack(fill="x", padx=10, pady=5)
dep_n_pulses_entry.insert(0, "50")

ctk.CTkLabel(synapse_cycle_frame, text="Dep Read Voltage (V):").pack(anchor="w", padx=10)
dep_read_voltage_entry = ctk.CTkEntry(synapse_cycle_frame)
dep_read_voltage_entry.pack(fill="x", padx=10, pady=5)
dep_read_voltage_entry.insert(0, "0.1")

synapse_cycle_button_frame = ctk.CTkFrame(synapse_cycle_frame)
synapse_cycle_button_frame.pack(fill="x", padx=10, pady=10)

run_cycle_button = ctk.CTkButton(synapse_cycle_button_frame, text="Run Pot-Dep Cycles", command=run_cycle_characterization, fg_color="#C62828", hover_color="#8E0000")
run_cycle_button.pack(fill="x", pady=5)

save_cycle_button = ctk.CTkButton(synapse_cycle_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_cycle_button.pack(fill="x", pady=5)

# --- MULTI-DEVICE SUBMODE ---
synapse_multidevice_frame = ctk.CTkFrame(synapse_frame)

ctk.CTkLabel(synapse_multidevice_frame, text="─ Multi-Device Parameters ─", font=("Arial", 10, "bold")).pack(pady=(10, 5))

multi_device_var = ctk.IntVar()
multi_device_check = ctk.CTkCheckBox(synapse_multidevice_frame, text="Enable Multi-Device Mode", variable=multi_device_var,
                                      command=toggle_multi_device_mode)
multi_device_check.pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(synapse_multidevice_frame, text="# Devices:").pack(anchor="w", padx=10)
n_devices_entry = ctk.CTkEntry(synapse_multidevice_frame)
n_devices_entry.pack(fill="x", padx=10, pady=5)
n_devices_entry.insert(0, "2")
n_devices_entry.configure(state="disabled")

ctk.CTkLabel(synapse_multidevice_frame, text="Inter-device Delay (ms):").pack(anchor="w", padx=10)
inter_device_delay_entry = ctk.CTkEntry(synapse_multidevice_frame)
inter_device_delay_entry.pack(fill="x", padx=10, pady=5)
inter_device_delay_entry.insert(0, "2000")
inter_device_delay_entry.configure(state="disabled")

ctk.CTkLabel(synapse_multidevice_frame, text="Device Name Pattern:").pack(anchor="w", padx=10)
device_name_pattern_entry = ctk.CTkEntry(synapse_multidevice_frame)
device_name_pattern_entry.pack(fill="x", padx=10, pady=5)
device_name_pattern_entry.insert(0, "Device")
device_name_pattern_entry.configure(state="disabled")

ctk.CTkLabel(synapse_multidevice_frame, text="(Uses Cycle parameters above)", font=("Arial", 9, "italic")).pack(pady=5)

synapse_multidevice_button_frame = ctk.CTkFrame(synapse_multidevice_frame)
synapse_multidevice_button_frame.pack(fill="x", padx=10, pady=10)

run_multidevice_button = ctk.CTkButton(synapse_multidevice_button_frame, text="Run Multi-Device", command=run_cycle_characterization, fg_color="#C62828", hover_color="#8E0000")
run_multidevice_button.pack(fill="x", pady=5)

save_multidevice_button = ctk.CTkButton(synapse_multidevice_button_frame, text="Save Data", command=save_synapse_data, fg_color="#1565C0", hover_color="#0D47A1")
save_multidevice_button.pack(fill="x", pady=5)

# ==============================================================================
# GRAPH SETUP
# ==============================================================================

plt.ioff()  # Turn off interactive mode to prevent empty window
fig, ax = plt.subplots(figsize=(7, 3))
ax.set_title("JV Curve")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current Density (mA/cm²)")
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# ==============================================================================
# MODE DISPLAY LOGIC
# ==============================================================================

def update_mode_display():
    """Show/hide parameter frames based on selected mode."""
    mode = mode_selection.get()
    
    # Hide all mode frames
    diode_frame.pack_forget()
    transistor_frame.pack_forget()
    synapse_frame.pack_forget()
    
    # Show selected mode frame
    if mode == "Diode":
        diode_frame.pack(fill="both", expand=True, pady=10)
    elif mode == "Transistor":
        transistor_frame.pack(fill="both", expand=True, pady=10)
        # Link transistor entries to diode entries for shared parameters
        start_voltage_trans.delete(0, 'end')
        start_voltage_trans.insert(0, start_voltage.get() if start_voltage.get() else "")
        end_voltage_trans.delete(0, 'end')
        end_voltage_trans.insert(0, end_voltage.get() if end_voltage.get() else "")
        voltage_step_trans.delete(0, 'end')
        voltage_step_trans.insert(0, voltage_step.get() if voltage_step.get() else "")
    elif mode == "Synapse":
        synapse_frame.pack(fill="both", expand=True, pady=10)
        update_synapse_submode()

def update_synapse_submode():
    """Show/hide synapse parameter frames based on selected submode."""
    submode = synapse_submode.get()
    
    # Hide all synapse submode frames
    synapse_basic_frame.pack_forget()
    synapse_srdp_frame.pack_forget()
    synapse_stdp_frame.pack_forget()
    synapse_cycle_frame.pack_forget()
    synapse_multidevice_frame.pack_forget()
    
    # Show selected submode frame
    if submode == "Basic":
        synapse_basic_frame.pack(fill="both", expand=True, pady=5)
    elif submode == "SRDP":
        synapse_srdp_frame.pack(fill="both", expand=True, pady=5)
    elif submode == "STDP":
        synapse_stdp_frame.pack(fill="both", expand=True, pady=5)
    elif submode == "Cycle":
        synapse_cycle_frame.pack(fill="both", expand=True, pady=5)
    elif submode == "Multi-Device":
        synapse_cycle_frame.pack(fill="both", expand=True, pady=5)
        synapse_multidevice_frame.pack(fill="both", expand=True, pady=5)

# ==============================================================================
# HELPER FUNCTIONS FOR TRANSISTOR MODE
# ==============================================================================

def sync_transistor_to_diode():
    """Copy diode voltage parameters to transistor before measurement."""
    global start_voltage, end_voltage, voltage_step
    # Create references that point to transistor entries
    start_voltage = start_voltage_trans
    end_voltage = end_voltage_trans
    voltage_step = voltage_step_trans

# ==============================================================================
# MODIFIED RUN FUNCTIONS TO HANDLE PARAMETER SOURCES
# ==============================================================================

# Wrap run_transistor_measurement to sync parameters first
original_run_transistor = run_transistor_measurement
def run_transistor_measurement():
    sync_transistor_to_diode()
    original_run_transistor()

# Wrap run_srdp_characterization to use correct stim parameters
original_run_srdp = run_srdp_characterization
def run_srdp_characterization():
    # Temporarily update stim_level_entry reference for SRDP
    global stim_level_entry, stim_width_entry, n_pulses_entry
    old_stim_level = stim_level_entry
    old_stim_width = stim_width_entry
    old_n_pulses = n_pulses_entry
    
    stim_level_entry = srdp_stim_level_entry
    stim_width_entry = srdp_stim_width_entry
    n_pulses_entry = srdp_n_pulses_entry
    
    try:
        original_run_srdp()
    finally:
        stim_level_entry = old_stim_level
        stim_width_entry = old_stim_width
        n_pulses_entry = old_n_pulses


# Initialize display to show Diode mode
update_mode_display()

# ==============================================================================
# WATERMARK
# ==============================================================================

watermark_label = ctk.CTkLabel(root, text="Zacharie Jehl Li-Kao --- zacharie.jehl@upc.edu", text_color="gray")
watermark_label.pack(side="bottom", anchor="se", padx=10, pady=10)

# ==============================================================================
# START APPLICATION
# ==============================================================================

root.mainloop()