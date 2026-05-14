import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from generator import generate_synthetic_and_metrics
from PIL import Image
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: white;
        color: black;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: white;
    }

    /* Text color */
    html, body, [class*="css"] {
        color: black;
    }

    /* Titles */
    h1, h2, h3 {
        color: black;
    }

    /* Buttons */
    .stButton>button {
        background-color: #f0f0f0;
        color: black;
        border-radius: 8px;
        border: 1px solid #ccc;
    }

    /* Metric boxes */
    [data-testid="metric-container"] {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Welding Signal Dashboard", layout="wide")
# ===============================
# HEADER WITH LOGOS
# ===============================
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.image("iit_logo.png", width=100)

with col2:
    st.markdown(
        "<h1 style='text-align: center;'>🔥 Welding Signal Synthetic Evaluation Dashboard</h1>",
        unsafe_allow_html=True
    )

with col3:
    st.image("mech_logo.png", width=100)

#st.title("🔥 Welding Signal Synthetic Evaluation Dashboard")

st.markdown("This dashboard compares real and synthetic welding signals using statistical and signal-processing metrics.")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    start = st.number_input("Start Row", 0, len(df)-1, 0)
    end = st.number_input("End Row", 1, len(df), min(1000, len(df)))

    if st.button("Generate & Evaluate"):

        result = generate_synthetic_and_metrics(df, start, end)

        if len(result) != 12:
            st.error(f"Expected 12 outputs, got {len(result)}")
            st.stop()
   
        (x_real, syn, t_real,
         f_real, psd_real,
         f_syn, psd_syn,
         acf_real, acf_syn,
         power_real, power_syn,
         metrics) = result

        # ===============================
        # CURRENT SIGNAL
        # ===============================
        st.subheader("📈 Current Signal Comparison")
        st.markdown("""
        This graph compares the **real welding current signal** with the **synthetic generated signal** over time.  
        It helps evaluate how well the synthetic model replicates amplitude variations and switching behavior.
        """)

        fig1, ax1 = plt.subplots(figsize=(12,4))
        ax1.plot(x_real, label="Real Current")
        ax1.plot(syn, label="Synthetic Current")
        ax1.set_title("Current vs Time")
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Current (A)")
        ax1.legend()
        st.pyplot(fig1)

        # ===============================
        # POWER
        # ===============================
        st.subheader("⚡ Power Signal Comparison (P = V × I)")
        st.markdown("""
        Power is computed assuming a constant voltage (25V).  
        This graph shows how energy delivery varies over time in both real and synthetic signals.  
        It is crucial for analyzing welding performance and heat input.
        """)

        fig2, ax2 = plt.subplots(figsize=(12,4))
        ax2.plot(power_real, label="Real Power")
        ax2.plot(power_syn, label="Synthetic Power")
        ax2.set_title("Power vs Time")
        ax2.set_xlabel("Samples")
        ax2.set_ylabel("Power (W)")
        ax2.legend()
        st.pyplot(fig2)
        
        # ===============================
        # AVERAGE POWER
        # ===============================
        st.subheader("📊 Average Power Comparison")
        st.markdown("""
        Average power represents the mean energy delivery over the selected range.  
        It helps compare overall intensity of real vs synthetic welding process.
        """)

        avg_power_real = power_real.mean()
        avg_power_syn  = power_syn.mean()

        fig_avg, ax_avg = plt.subplots(figsize=(6,4))
        ax_avg.bar(["Real", "Synthetic"], [avg_power_real, avg_power_syn])
        ax_avg.set_title("Average Power Comparison")
        ax_avg.set_ylabel("Power (W)")

        st.pyplot(fig_avg)

        st.write(f"🔹 Real Avg Power: {avg_power_real:.2f} W")
        st.write(f"🔹 Synthetic Avg Power: {avg_power_syn:.2f} W")
        # ===============================
        # CUMULATIVE POWER (ENERGY)
        # ===============================
        st.subheader("⚡ Cumulative Energy (∫P dt)")
        st.markdown("""
        Cumulative energy shows total energy delivered over time.  
        It is critical for understanding heat input in welding processes.
        """)

        # time step (convert ms → sec)
        dt = (t_real[1] - t_real[0]) / 1000 if len(t_real) > 1 else 1

        energy_real = (power_real * dt).cumsum()
        energy_syn  = (power_syn * dt).cumsum()

        fig_energy, ax_energy = plt.subplots(figsize=(12,4))
        ax_energy.plot(energy_real, label="Real Energy")
        ax_energy.plot(energy_syn, label="Synthetic Energy")
        ax_energy.set_title("Cumulative Energy vs Time")
        ax_energy.set_xlabel("Samples")
        ax_energy.set_ylabel("Energy (J)")
        ax_energy.legend()

        st.pyplot(fig_energy)

        # ===============================
        # METRICS
        # ===============================
        st.subheader("📊 Quantitative Metrics")
        st.markdown("""
        These metrics quantify similarity between real and synthetic signals:
        - **RMS Error** → amplitude similarity  
        - **PSD Difference** → frequency content similarity  
        - **ACF Correlation** → temporal pattern similarity  
        - **Kurtosis Error** → distribution shape similarity  
        """)

        cols = st.columns(4)
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(k, f"{v:.3f}")

        # ===============================
        # PSD
        # ===============================
        st.subheader("📡 Power Spectral Density (Frequency Analysis)")
        st.markdown("""
        PSD shows how signal power is distributed across frequencies.  
        Matching PSD indicates that synthetic signals preserve frequency characteristics of the real process.
        """)

        fig3, ax3 = plt.subplots(figsize=(12,4))
        ax3.semilogy(f_real, psd_real, label="Real")
        ax3.semilogy(f_syn, psd_syn, label="Synthetic")
        ax3.set_title("PSD Comparison")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power/Frequency")
        ax3.legend()
        st.pyplot(fig3)

        # ===============================
        # ACF
        # ===============================
        st.subheader("🔁 Autocorrelation Function (Temporal Dependency)")
        st.markdown("""
        ACF measures how the signal correlates with itself over time lags.  
        A high similarity indicates that temporal dynamics and switching patterns are preserved.
        """)

        fig4, ax4 = plt.subplots(figsize=(12,4))
        ax4.plot(acf_real, label="Real")
        ax4.plot(acf_syn, label="Synthetic")
        ax4.set_title("ACF Comparison")
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("Correlation")
        ax4.legend()
        st.pyplot(fig4)