import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyvisa
import time
from tkinter.filedialog import asksaveasfilename
import synapse_engine

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
        
        # Collect base parameters
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different.")
        
        base_params = {
            "mode": "srdp",
            "stim_drive_type": stim_drive_var.get(),
            "stim_level": validate_float(stim_level_entry.get(), "Stim Level"),
            "stim_width_ms": validate_float(stim_width_entry.get(), "Stim Width"),
            "n_pulses": int(validate_float(n_pulses_entry.get(), "# Pulses")),
            "read_voltage": validate_float(read_voltage_entry.get(), "Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5
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
        
        # Collect base parameters
        stim_ch_str = stim_channel_var.get().lower()
        read_ch_str = read_channel_var.get().lower()
        
        if stim_ch_str == read_ch_str:
            raise ValueError("Stim and Read channels must be different.")
        
        base_params = {
            "mode": "stdp",
            "stim_drive_type": "V",  # STDP typically uses voltage
            "stim_level": validate_float(stim_level_entry.get(), "Pre-spike Level"),
            "post_spike_level": validate_float(stdp_post_level_entry.get(), "Post-spike Level"),
            "stim_width_ms": validate_float(stim_width_entry.get(), "Pulse Width"),
            "stim_period_ms": validate_float(stim_period_entry.get(), "Pair Period"),
            "n_pulses": int(validate_float(stdp_n_pairs_entry.get(), "# Spike Pairs")),
            "read_voltage": validate_float(read_voltage_entry.get(), "Read Voltage"),
            "read_delay_ms": validate_float(read_delay_entry.get(), "Read Delay"),
            "compliance_A": validate_float(compliance_synapse_entry.get(), "Compliance"),
            "settle_ms": 5
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
            "settle_ms": 5
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

# Create main application window
root = ctk.CTk()
root.title("Shockingly Accurate IV")
root.geometry("1400x700")

# Left Frame for Inputs
input_frame = ctk.CTkScrollableFrame(root, width=300)
input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Graph Frame
graph_frame = ctk.CTkFrame(root)
graph_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Address Selection
ctk.CTkLabel(input_frame, text="Connection Type:").grid(row=0, column=0, sticky="w", pady=5)
connection_type = ctk.CTkComboBox(input_frame, values=["GPIB", "RS232", "LAN"])
connection_type.grid(row=0, column=1, sticky="ew", pady=5)
connection_type.set("GPIB")

ctk.CTkLabel(input_frame, text="Port Address:").grid(row=1, column=0, sticky="w", pady=5)
port_entry = ctk.CTkEntry(input_frame)
port_entry.grid(row=1, column=1, sticky="ew", pady=5)

# Voltage Range
ctk.CTkLabel(input_frame, text="Starting Voltage (V):").grid(row=2, column=0, sticky="w", pady=5)
start_voltage = ctk.CTkEntry(input_frame)
start_voltage.grid(row=2, column=1, sticky="ew", pady=5)

ctk.CTkLabel(input_frame, text="Ending Voltage (V):").grid(row=3, column=0, sticky="w", pady=5)
end_voltage = ctk.CTkEntry(input_frame)
end_voltage.grid(row=3, column=1, sticky="ew", pady=5)

ctk.CTkLabel(input_frame, text="Voltage Step (V):").grid(row=4, column=0, sticky="w", pady=5)
voltage_step = ctk.CTkEntry(input_frame)
voltage_step.grid(row=4, column=1, sticky="ew", pady=5)

# Measurement Speed
ctk.CTkLabel(input_frame, text="Measurement Speed (NPLC):").grid(row=5, column=0, sticky="w", pady=5)
nplc_entry = ctk.CTkEntry(input_frame)
nplc_entry.grid(row=5, column=1, sticky="ew", pady=5)

# Timeout Duration
ctk.CTkLabel(input_frame, text="Timeout Duration (s):").grid(row=6, column=0, sticky="w", pady=5)
timeout_entry = ctk.CTkEntry(input_frame)
timeout_entry.grid(row=6, column=1, sticky="ew", pady=5)
timeout_entry.insert(0, "30")

# Compliance
ctk.CTkLabel(input_frame, text="Compliance (A):").grid(row=7, column=0, sticky="w", pady=5)
compliance = ctk.CTkComboBox(input_frame, values=["100nA", "1µA", "10µA", "100µA", "1mA", "10mA", "100mA", "1A", "1.5A"])
compliance.grid(row=7, column=1, sticky="ew", pady=5)
compliance.set("1mA")

# Channel Selection
ctk.CTkLabel(input_frame, text="Channel:").grid(row=8, column=0, sticky="w", pady=5)
channel_selection = ctk.CTkComboBox(input_frame, values=["Channel A", "Channel B"])
channel_selection.grid(row=8, column=1, sticky="ew", pady=5)
channel_selection.set("Channel A")

# Cell #
ctk.CTkLabel(input_frame, text="Cell #:").grid(row=9, column=0, sticky="w", pady=5)
sample_name_entry = ctk.CTkEntry(input_frame)
sample_name_entry.grid(row=9, column=1, sticky="ew", pady=5)

# Sample Surface Area
ctk.CTkLabel(input_frame, text="Cell Surface Area (cm²):").grid(row=10, column=0, sticky="w", pady=5)
surface_area = ctk.CTkEntry(input_frame)
surface_area.grid(row=10, column=1, sticky="ew", pady=5)

# Hysteresis Measurement
hysteresis_var = ctk.IntVar()
hysteresis_check = ctk.CTkCheckBox(input_frame, text="Perform Hysteresis", variable=hysteresis_var)
hysteresis_check.grid(row=11, column=0, columnspan=2, sticky="w", pady=5)

ctk.CTkLabel(input_frame, text="Hysteresis Cycles:").grid(row=12, column=0, sticky="w", pady=5)
hysteresis_cycles = ctk.CTkEntry(input_frame)
hysteresis_cycles.grid(row=12, column=1, sticky="ew", pady=5)

# Dark Measurement Checkbox
dark_measurement = ctk.IntVar()
dark_check = ctk.CTkCheckBox(input_frame, text="Dark JV", variable=dark_measurement)
dark_check.grid(row=13, column=0, columnspan=2, sticky="w", pady=5)

# Irradiance Input
ctk.CTkLabel(input_frame, text="Irradiance (W/m²):").grid(row=14, column=0, sticky="w", pady=5)
light_power_entry = ctk.CTkEntry(input_frame)
light_power_entry.grid(row=14, column=1, sticky="ew", pady=5)
light_power_entry.insert(0, "1000")

# Wire mode selection
wire_mode = ctk.CTkComboBox(input_frame, values=["2-Wire", "4-Wire"])
wire_mode.grid(row=15, column=0, columnspan=2, sticky="ew", pady=5)
wire_mode.set("4-Wire")

# Autorange Toggle
autorange_var = ctk.IntVar(value=0)
autorange_check = ctk.CTkCheckBox(input_frame, text="Enable Autorange", variable=autorange_var)
autorange_check.grid(row=16, column=0, columnspan=2, sticky="w", pady=5)

# Transistor Mode Controls
ctk.CTkLabel(input_frame, text="Gate Start Voltage (V):").grid(row=17, column=0, sticky="w", pady=5)
gate_start_voltage = ctk.CTkEntry(input_frame)
gate_start_voltage.grid(row=17, column=1, sticky="ew", pady=5)

ctk.CTkLabel(input_frame, text="Gate End Voltage (V):").grid(row=18, column=0, sticky="w", pady=5)
gate_end_voltage = ctk.CTkEntry(input_frame)
gate_end_voltage.grid(row=18, column=1, sticky="ew", pady=5)

ctk.CTkLabel(input_frame, text="Gate Voltage Step (V):").grid(row=19, column=0, sticky="w", pady=5)
gate_voltage_step = ctk.CTkEntry(input_frame)
gate_voltage_step.grid(row=19, column=1, sticky="ew", pady=5)

# Mode Selector
ctk.CTkLabel(input_frame, text="Measurement Mode:").grid(row=20, column=0, sticky="w", pady=5)
mode_selection = ctk.CTkComboBox(input_frame, values=["Diode", "Transistor"], command=toggle_mode)
mode_selection.grid(row=20, column=1, sticky="ew", pady=5)
mode_selection.set("Diode")

# Initially hide transistor controls
gate_start_voltage.grid_remove()
gate_end_voltage.grid_remove()
gate_voltage_step.grid_remove()


# ==============================================================================
# SYNAPSE / MEMRISTOR CONTROLS
# ==============================================================================

# Separator
ctk.CTkLabel(input_frame, text="─── Synapse / Memristor ───", font=("Arial", 12, "bold")).grid(
    row=21, column=0, columnspan=2, pady=(10, 5)
)

# Mode Selection for Synapse
ctk.CTkLabel(input_frame, text="Synapse Mode:").grid(row=22, column=0, sticky="w", pady=5)
synapse_mode_selection = ctk.CTkComboBox(
    input_frame, 
    values=["Electrical", "Visual", "Memristor (Pulse)"]
)
synapse_mode_selection.grid(row=22, column=1, sticky="ew", pady=5)
synapse_mode_selection.set("Electrical")

# Stim Channel
ctk.CTkLabel(input_frame, text="Stim Channel:").grid(row=23, column=0, sticky="w", pady=5)
stim_channel_var = ctk.CTkComboBox(input_frame, values=["smua", "smub"])
stim_channel_var.grid(row=23, column=1, sticky="ew", pady=5)
stim_channel_var.set("smua")

# Read Channel
ctk.CTkLabel(input_frame, text="Read Channel:").grid(row=24, column=0, sticky="w", pady=5)
read_channel_var = ctk.CTkComboBox(input_frame, values=["smua", "smub"])
read_channel_var.grid(row=24, column=1, sticky="ew", pady=5)
read_channel_var.set("smub")

# Stim Drive Type
ctk.CTkLabel(input_frame, text="Stim Drive Type:").grid(row=25, column=0, sticky="w", pady=5)
stim_drive_var = ctk.CTkComboBox(input_frame, values=["V", "I"])
stim_drive_var.grid(row=25, column=1, sticky="ew", pady=5)
stim_drive_var.set("V")

# Stim Level
ctk.CTkLabel(input_frame, text="Stim Level (V or A):").grid(row=26, column=0, sticky="w", pady=5)
stim_level_entry = ctk.CTkEntry(input_frame)
stim_level_entry.grid(row=26, column=1, sticky="ew", pady=5)
stim_level_entry.insert(0, "1.0")

# Stim Width
ctk.CTkLabel(input_frame, text="Stim Width (ms):").grid(row=27, column=0, sticky="w", pady=5)
stim_width_entry = ctk.CTkEntry(input_frame)
stim_width_entry.grid(row=27, column=1, sticky="ew", pady=5)
stim_width_entry.insert(0, "100")

# Stim Period
ctk.CTkLabel(input_frame, text="Stim Period (ms):").grid(row=28, column=0, sticky="w", pady=5)
stim_period_entry = ctk.CTkEntry(input_frame)
stim_period_entry.grid(row=28, column=1, sticky="ew", pady=5)
stim_period_entry.insert(0, "200")

# Number of Pulses
ctk.CTkLabel(input_frame, text="# Pulses:").grid(row=29, column=0, sticky="w", pady=5)
n_pulses_entry = ctk.CTkEntry(input_frame)
n_pulses_entry.grid(row=29, column=1, sticky="ew", pady=5)
n_pulses_entry.insert(0, "50")

# Read Voltage
ctk.CTkLabel(input_frame, text="Read Voltage (V):").grid(row=30, column=0, sticky="w", pady=5)
read_voltage_entry = ctk.CTkEntry(input_frame)
read_voltage_entry.grid(row=30, column=1, sticky="ew", pady=5)
read_voltage_entry.insert(0, "0.1")

# Read Delay
ctk.CTkLabel(input_frame, text="Read Delay (ms):").grid(row=31, column=0, sticky="w", pady=5)
read_delay_entry = ctk.CTkEntry(input_frame)
read_delay_entry.grid(row=31, column=1, sticky="ew", pady=5)
read_delay_entry.insert(0, "10")

# Compliance (Synapse)
ctk.CTkLabel(input_frame, text="Compliance (A):").grid(row=32, column=0, sticky="w", pady=5)
compliance_synapse_entry = ctk.CTkEntry(input_frame)
compliance_synapse_entry.grid(row=32, column=1, sticky="ew", pady=5)
compliance_synapse_entry.insert(0, "100e-6")

# Simulation Mode Checkbox
simulate_var = ctk.IntVar()
simulate_check = ctk.CTkCheckBox(
    input_frame, 
    text="Simulate (No Hardware)", 
    variable=simulate_var
)
simulate_check.grid(row=33, column=0, columnspan=2, sticky="w", pady=5)


# Parameter Presets
ctk.CTkLabel(input_frame, text="Presets:").grid(row=34, column=0, sticky="w", pady=5)
preset_var = ctk.CTkComboBox(
    input_frame, 
    values=["Custom", "LTP Moderate", "LTD Moderate", "PPF Test", "High Speed"],
    command=lambda choice: apply_preset(choice)
)
preset_var.grid(row=34, column=1, sticky="ew", pady=5)
preset_var.set("Custom")

# ==============================================================================
# SRDP CHARACTERIZATION CONTROLS
# ==============================================================================

ctk.CTkLabel(input_frame, text="─── SRDP Characterization ───", font=("Arial", 12, "bold")).grid(
    row=35, column=0, columnspan=2, pady=(10, 5)
)

ctk.CTkLabel(input_frame, text="Freq Start (Hz):").grid(row=36, column=0, sticky="w", pady=5)
srdp_freq_start_entry = ctk.CTkEntry(input_frame)
srdp_freq_start_entry.grid(row=36, column=1, sticky="ew", pady=5)
srdp_freq_start_entry.insert(0, "1")

ctk.CTkLabel(input_frame, text="Freq End (Hz):").grid(row=37, column=0, sticky="w", pady=5)
srdp_freq_end_entry = ctk.CTkEntry(input_frame)
srdp_freq_end_entry.grid(row=37, column=1, sticky="ew", pady=5)
srdp_freq_end_entry.insert(0, "100")

ctk.CTkLabel(input_frame, text="# Freq Points:").grid(row=38, column=0, sticky="w", pady=5)
srdp_freq_points_entry = ctk.CTkEntry(input_frame)
srdp_freq_points_entry.grid(row=38, column=1, sticky="ew", pady=5)
srdp_freq_points_entry.insert(0, "10")

srdp_log_scale_var = ctk.IntVar(value=1)
srdp_log_scale_check = ctk.CTkCheckBox(
    input_frame, 
    text="Log Scale Frequency", 
    variable=srdp_log_scale_var
)
srdp_log_scale_check.grid(row=39, column=0, columnspan=2, sticky="w", pady=5)

# ==============================================================================
# STDP CHARACTERIZATION CONTROLS
# ==============================================================================

ctk.CTkLabel(input_frame, text="─── STDP Characterization ───", font=("Arial", 12, "bold")).grid(
    row=40, column=0, columnspan=2, pady=(10, 5)
)

ctk.CTkLabel(input_frame, text="Δt Start (ms):").grid(row=41, column=0, sticky="w", pady=5)
stdp_dt_start_entry = ctk.CTkEntry(input_frame)
stdp_dt_start_entry.grid(row=41, column=1, sticky="ew", pady=5)
stdp_dt_start_entry.insert(0, "-50")

ctk.CTkLabel(input_frame, text="Δt End (ms):").grid(row=42, column=0, sticky="w", pady=5)
stdp_dt_end_entry = ctk.CTkEntry(input_frame)
stdp_dt_end_entry.grid(row=42, column=1, sticky="ew", pady=5)
stdp_dt_end_entry.insert(0, "50")

ctk.CTkLabel(input_frame, text="# Δt Points:").grid(row=43, column=0, sticky="w", pady=5)
stdp_dt_points_entry = ctk.CTkEntry(input_frame)
stdp_dt_points_entry.grid(row=43, column=1, sticky="ew", pady=5)
stdp_dt_points_entry.insert(0, "15")

ctk.CTkLabel(input_frame, text="# Spike Pairs:").grid(row=44, column=0, sticky="w", pady=5)
stdp_n_pairs_entry = ctk.CTkEntry(input_frame)
stdp_n_pairs_entry.grid(row=44, column=1, sticky="ew", pady=5)
stdp_n_pairs_entry.insert(0, "50")

ctk.CTkLabel(input_frame, text="Post-spike Level (V):").grid(row=45, column=0, sticky="w", pady=5)
stdp_post_level_entry = ctk.CTkEntry(input_frame)
stdp_post_level_entry.grid(row=45, column=1, sticky="ew", pady=5)
stdp_post_level_entry.insert(0, "1.0")

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


# ==============================================================================
# SRDP CHARACTERIZATION CONTROLS
# ==============================================================================

ctk.CTkLabel(input_frame, text="─── SRDP Characterization ───", font=("Arial", 12, "bold")).grid(
    row=35, column=0, columnspan=2, pady=(10, 5)
)

ctk.CTkLabel(input_frame, text="Freq Start (Hz):").grid(row=36, column=0, sticky="w", pady=5)
srdp_freq_start_entry = ctk.CTkEntry(input_frame)
srdp_freq_start_entry.grid(row=36, column=1, sticky="ew", pady=5)
srdp_freq_start_entry.insert(0, "1")

ctk.CTkLabel(input_frame, text="Freq End (Hz):").grid(row=37, column=0, sticky="w", pady=5)
srdp_freq_end_entry = ctk.CTkEntry(input_frame)
srdp_freq_end_entry.grid(row=37, column=1, sticky="ew", pady=5)
srdp_freq_end_entry.insert(0, "100")

ctk.CTkLabel(input_frame, text="# Freq Points:").grid(row=38, column=0, sticky="w", pady=5)
srdp_freq_points_entry = ctk.CTkEntry(input_frame)
srdp_freq_points_entry.grid(row=38, column=1, sticky="ew", pady=5)
srdp_freq_points_entry.insert(0, "10")

srdp_log_scale_var = ctk.IntVar(value=1)
srdp_log_scale_check = ctk.CTkCheckBox(
    input_frame, 
    text="Log Scale Frequency", 
    variable=srdp_log_scale_var
)
srdp_log_scale_check.grid(row=39, column=0, columnspan=2, sticky="w", pady=5)

# ==============================================================================
# STDP CHARACTERIZATION CONTROLS
# ==============================================================================

ctk.CTkLabel(input_frame, text="─── STDP Characterization ───", font=("Arial", 12, "bold")).grid(
    row=40, column=0, columnspan=2, pady=(10, 5)
)

ctk.CTkLabel(input_frame, text="Δt Start (ms):").grid(row=41, column=0, sticky="w", pady=5)
stdp_dt_start_entry = ctk.CTkEntry(input_frame)
stdp_dt_start_entry.grid(row=41, column=1, sticky="ew", pady=5)
stdp_dt_start_entry.insert(0, "-50")

ctk.CTkLabel(input_frame, text="Δt End (ms):").grid(row=42, column=0, sticky="w", pady=5)
stdp_dt_end_entry = ctk.CTkEntry(input_frame)
stdp_dt_end_entry.grid(row=42, column=1, sticky="ew", pady=5)
stdp_dt_end_entry.insert(0, "50")

ctk.CTkLabel(input_frame, text="# Δt Points:").grid(row=43, column=0, sticky="w", pady=5)
stdp_dt_points_entry = ctk.CTkEntry(input_frame)
stdp_dt_points_entry.grid(row=43, column=1, sticky="ew", pady=5)
stdp_dt_points_entry.insert(0, "15")

ctk.CTkLabel(input_frame, text="# Spike Pairs:").grid(row=44, column=0, sticky="w", pady=5)
stdp_n_pairs_entry = ctk.CTkEntry(input_frame)
stdp_n_pairs_entry.grid(row=44, column=1, sticky="ew", pady=5)
stdp_n_pairs_entry.insert(0, "50")

ctk.CTkLabel(input_frame, text="Post-spike Level (V):").grid(row=45, column=0, sticky="w", pady=5)
stdp_post_level_entry = ctk.CTkEntry(input_frame)
stdp_post_level_entry.grid(row=45, column=1, sticky="ew", pady=5)
stdp_post_level_entry.insert(0, "1.0")

# Parameters Display Frame
params_frame = ctk.CTkFrame(graph_frame)
params_frame.pack(side="right", fill="y", padx=5)
params_text = ctk.CTkTextbox(params_frame, width=200, height=400)
params_text.pack(pady=10, padx=5)

# Graph Setup
fig, ax = plt.subplots(figsize=(7, 2.5))
ax.set_title("JV Curve")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current Density (mA/cm²)")
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# Button Frame
button_frame = ctk.CTkFrame(root)
button_frame.grid(row=1, column=0, columnspan=2, pady=10)

# Run Button
run_button = ctk.CTkButton(button_frame, text="Run Measurement", command=run_measurement_buffered)
run_button.pack(side="left", padx=10)

# Save Button
save_button = ctk.CTkButton(button_frame, text="Save JV Curve", command=save_curve)
save_button.pack(side="left", padx=10)

# Transistor Measurement Button
transistor_button = ctk.CTkButton(button_frame, text="Run Transistor Measurement", command=run_transistor_measurement)
transistor_button.pack(side="left", padx=10)

# Save Transistor Data Button
save_transistor_button = ctk.CTkButton(button_frame, text="Save Transistor Data", command=save_transistor_data)
save_transistor_button.pack(side="left", padx=10)

# Synapse Measurement Button
synapse_button = ctk.CTkButton(
    button_frame, 
    text="Run Synapse Measurement", 
    command=run_synapse_mode,
    fg_color="#2E7D32",  # Green color
    hover_color="#1B5E20"
)
synapse_button.pack(side="left", padx=10)

# Save Synapse Data Button
save_synapse_button = ctk.CTkButton(
    button_frame, 
    text="Save Synapse Data", 
    command=save_synapse_data,
    fg_color="#1565C0",  # Blue color
    hover_color="#0D47A1"
)
save_synapse_button.pack(side="left", padx=10)

# SRDP Characterization Button
srdp_button = ctk.CTkButton(
    button_frame, 
    text="Run SRDP Characterization", 
    command=run_srdp_characterization,
    fg_color="#FF6F00",  # Orange color
    hover_color="#E65100"
)
srdp_button.pack(side="left", padx=10)

# STDP Characterization Button
stdp_button = ctk.CTkButton(
    button_frame, 
    text="Run STDP Characterization", 
    command=run_stdp_characterization,
    fg_color="#7B1FA2",  # Purple color
    hover_color="#4A148C"
)
stdp_button.pack(side="left", padx=10)


# Watermark
watermark_label = ctk.CTkLabel(root, text="Zacharie Jehl Li-Kao --- zacharie.jehl@upc.edu", text_color="gray")
watermark_label.grid(row=2, column=0, sticky="sw", padx=10, pady=10)

# Configure column and row weights
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(0, weight=1)

# Start the application
root.mainloop()