import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pyvisa
import time
from tkinter.filedialog import asksaveasfilename

# appearance mode and color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Arrays to store JV curves and PV parameters
jv_curves = []  # To store [(sample_name, voltages, current_density)]
pv_parameters = []  # To store [(sample_name, parameters_dict)]

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

# Watermark
watermark_label = ctk.CTkLabel(root, text="Zacharie Jehl Li-Kao --- zacharie.jehl@upc.edu", text_color="gray")
watermark_label.grid(row=2, column=0, sticky="sw", padx=10, pady=10)

# Configure column and row weights
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)
root.grid_rowconfigure(0, weight=1)

# Start the application
root.mainloop()