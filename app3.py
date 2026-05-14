import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from gen2 import generate_synthetic_and_metrics

st.title("🔥 Welding Signal Synthetic Evaluation Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    max_rows = len(df)

    start_row = st.number_input("Start Row", 0, max_rows-1, 0)
    end_row   = st.number_input("End Row", 1, max_rows, 1000)

    if st.button("Generate & Evaluate"):

        (x_real, syn, t_real,
         f_real, psd_real, f_syn, psd_syn,
         acf_real, acf_syn,
         power_real, power_syn,   # 👈 NEW
         metrics) = generate_synthetic_and_metrics(df, start_row, end_row)

        # ===============================
        # SIGNAL PLOT
        # ===============================
        fig1, ax1 = plt.subplots(figsize=(12,4))
        ax1.plot(x_real, label="Real")
        ax1.plot(syn, label="Synthetic")
        ax1.legend()
        ax1.set_title("Real vs Synthetic Current")
        st.pyplot(fig1)

        # ===============================
        # 🔥 POWER PLOT (NEW)
        # ===============================
        st.subheader("⚡ Power Comparison (P = 25 × I)")

        fig_power, axp = plt.subplots(figsize=(12,4))
        axp.plot(power_real, label="Real Power")
        axp.plot(power_syn, label="Synthetic Power")
        axp.set_title("Power vs Time")
        axp.legend()
        st.pyplot(fig_power)

        # ===============================
        # METRICS
        # ===============================
        st.subheader("📊 Core Metrics")
        for k, v in metrics.items():
            st.write(f"{k}: {v:.3f}")

        # ===============================
        # PSD
        # ===============================
        fig2, ax2 = plt.subplots(figsize=(12,4))
        ax2.semilogy(f_real, psd_real, label="Real")
        ax2.semilogy(f_syn, psd_syn, label="Synthetic")
        ax2.set_xlim(0, 50)
        ax2.set_title("PSD Comparison (0–50 Hz)")
        ax2.legend()
        st.pyplot(fig2)

        # ===============================
        # ACF
        # ===============================
        fig3, ax3 = plt.subplots(figsize=(10,4))
        ax3.plot(acf_real, label="Real")
        ax3.plot(acf_syn, label="Synthetic")
        ax3.set_title("ACF Comparison")
        ax3.legend()
        st.pyplot(fig3)