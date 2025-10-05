# --- Standard & plotting imports ---
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # optional
from io import BytesIO

import plotly.graph_objects as go
# import plotly.express as px  # optional
from plotly.subplots import make_subplots

# --- Make sure Python can import modules sitting next to this script ---
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- Import from quantileADF.py (must be in the same folder as this script) ---
try:
    # Adjust the imported names to exactly what quantileADF.py defines/what you use:
    from quantileADF import QADF, QAR, bootstraps
except Exception as e:
    st.error(f"Could not import from 'quantileADF.py'. "
             f"Make sure the file is in the same folder as this script. Details: {e}")
    st.stop()



# Page configuration
st.set_page_config(
    page_title="Quantile Unit Root Test",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 600;
    }
    h2 {
        color: #34495e;
        font-weight: 500;
    }
    h3 {
        color: #7f8c8d;
        font-weight: 500;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Quantile Unit Root Test (QADF)")
st.markdown("""
<div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='color: #1976d2; margin-top: 0;'>Koenker & Xiao (2004) Methodology</h3>
    <p style='color: #424242; margin-bottom: 0;'>
    This application performs quantile unit root tests to analyze the persistence of shocks
    across different quantiles of your time series data. Upload your data and explore
    stationarity properties beyond the mean.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File (.xlsx)",
    type=['xlsx', 'xls'],
    help="Upload your time series data in Excel format"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Load and preview data
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.session_state.data = df

        st.sidebar.success("✅ File uploaded successfully!")

        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            with col2:
                st.markdown("**Dataset Info:**")
                st.info(f"""
                - **Rows:** {len(df)}
                - **Columns:** {len(df.columns)}
                - **Memory:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB
                """)

        # Variable selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Variable Selection")

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_columns) > 0:
            selected_variable = st.sidebar.selectbox(
                "Select Time Series Variable",
                options=numeric_columns,
                help="Choose the variable for unit root testing"
            )

            # Model parameters
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔧 Model Parameters")

            model_type = st.sidebar.selectbox(
                "Model Type",
                options=['c', 'ct'],
                format_func=lambda x: 'Constant (c)' if x == 'c' else 'Constant + Trend (ct)',
                help="c: constant only, ct: constant and trend"
            )

            pmax = st.sidebar.slider(
                "Maximum Lags (pmax)",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of lags for serial correlation"
            )

            ic = st.sidebar.selectbox(
                "Information Criterion",
                options=['AIC', 'BIC', 't-stat'],
                help="Criterion for lag selection"
            )

            # Quantile settings
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Quantile Settings")

            quantile_option = st.sidebar.radio(
                "Quantile Selection",
                options=['Predefined', 'Custom Range', 'Single Quantile'],
                help="Choose how to specify quantiles"
            )

            if quantile_option == 'Predefined':
                quantiles = np.arange(0.1, 1.0, 0.1)
            elif quantile_option == 'Custom Range':
                col1, col2, col3 = st.sidebar.columns(3)
                q_start = col1.number_input("Start", 0.05, 0.95, 0.10, 0.05)
                q_end = col2.number_input("End", 0.05, 0.95, 0.90, 0.05)
                q_step = col3.number_input("Step", 0.01, 0.50, 0.10, 0.05)
                quantiles = np.arange(q_start, q_end + q_step, q_step)
            else:
                single_q = st.sidebar.slider("Quantile", 0.05, 0.95, 0.50, 0.05)
                quantiles = [single_q]

            # Display selected quantiles
            st.sidebar.info(f"**Selected quantiles:** {len(quantiles)}")

            # Run analysis button
            st.sidebar.markdown("---")
            run_analysis = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

            # Main analysis
            if run_analysis:
                with st.spinner('🔄 Running Quantile Unit Root Test...'):
                    try:
                        # Get the selected series
                        y = df[selected_variable].dropna()

                        # Initialize QADF model
                        qadf = QADF(endog=y, model=model_type, pmax=pmax, ic=ic)

                        # Fit for multiple quantiles
                        results_df = qadf.fitForQuantiles(quantiles)

                        # Store results
                        st.session_state.results = results_df
                        st.session_state.qadf = qadf
                        st.session_state.y = y
                        st.session_state.selected_variable = selected_variable

                        st.sidebar.success("✅ Analysis completed!")

                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.stop()

            # Display results if available
            if st.session_state.results is not None:
                results_df = st.session_state.results
                qadf = st.session_state.qadf
                y = st.session_state.y
                selected_variable = st.session_state.selected_variable

                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Results Summary",
                    "📈 Visualizations",
                    "🎯 Statistical Tests",
                    "🔍 Detailed Analysis",
                    "💾 Export"
                ])

                with tab1:
                    st.header("Results Summary")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #1976d2; margin: 0;'>QKS Statistic</h3>
                            <h2 style='margin: 10px 0;'>{:.3f}</h2>
                            <p style='color: #666; margin: 0;'>Quantile Kolmogorov-Smirnov</p>
                        </div>
                        """.format(results_df['QKS'].iloc[0]), unsafe_allow_html=True)

                    with col2:
                        avg_rho = results_df['ρ₁(τ)'].mean()
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #388e3c; margin: 0;'>Avg ρ₁(τ)</h3>
                            <h2 style='margin: 10px 0;'>{:.3f}</h2>
                            <p style='color: #666; margin: 0;'>Mean Reversion Speed</p>
                        </div>
                        """.format(avg_rho), unsafe_allow_html=True)

                    with col3:
                        lags_used = results_df['Lags'].iloc[0]
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #f57c00; margin: 0;'>Lags Used</h3>
                            <h2 style='margin: 10px 0;'>{}</h2>
                            <p style='color: #666; margin: 0;'>By {} Criterion</p>
                        </div>
                        """.format(lags_used, ic), unsafe_allow_html=True)

                    with col4:
                        avg_delta = results_df['δ²'].mean()
                        st.markdown("""
                        <div class='metric-card'>
                            <h3 style='color: #d32f2f; margin: 0;'>Avg δ²</h3>
                            <h2 style='margin: 10px 0;'>{:.3f}</h2>
                            <p style='color: #666; margin: 0;'>Nuisance Parameter</p>
                        </div>
                        """.format(avg_delta), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Results table
                    st.subheader("📋 Detailed Results Table")

                    # Color-code the table based on statistical significance
                    def highlight_significant(row):
                        colors = [''] * len(row)
                        stat = row['tₙ(τ)']
                        cv1 = row['CV1%']
                        cv5 = row['CV5%']
                        cv10 = row['CV10%']

                        if stat < cv1:
                            colors[7] = 'background-color: #c8e6c9'  # Green for 1% significance
                        elif stat < cv5:
                            colors[7] = 'background-color: #fff9c4'  # Yellow for 5% significance
                        elif stat < cv10:
                            colors[7] = 'background-color: #ffccbc'  # Orange for 10% significance

                        return colors

                    styled_results = results_df.style.apply(highlight_significant, axis=1)
                    st.dataframe(styled_results, use_container_width=True, height=400)

                    # Legend
                    st.markdown("""
                    **Legend:**
                    <span style='background-color: #c8e6c9; padding: 2px 8px; border-radius: 3px;'>Reject at 1%</span>
                    <span style='background-color: #fff9c4; padding: 2px 8px; border-radius: 3px;'>Reject at 5%</span>
                    <span style='background-color: #ffccbc; padding: 2px 8px; border-radius: 3px;'>Reject at 10%</span>
                    """, unsafe_allow_html=True)

                with tab2:
                    st.header("Visualizations")

                    # Time series plot
                    st.subheader("📈 Time Series Plot")
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        y=y,
                        mode='lines',
                        name=selected_variable,
                        line=dict(color='#1976d2', width=2)
                    ))
                    fig_ts.update_layout(
                        title=f"{selected_variable} Over Time",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        template="plotly_white",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

                    # Create subplots for quantile analysis
                    st.subheader("🎯 Quantile Analysis")

                    col1, col2 = st.columns(2)

                    with col1:
                        # ρ₁(τ) across quantiles
                        fig_rho = go.Figure()
                        fig_rho.add_trace(go.Scatter(
                            x=results_df.index,
                            y=results_df['ρ₁(τ)'],
                            mode='lines+markers',
                            name='ρ₁(τ)',
                            line=dict(color='#1976d2', width=3),
                            marker=dict(size=8)
                        ))
                        fig_rho.add_trace(go.Scatter(
                            x=results_df.index,
                            y=results_df['ρ₁(OLS)'],
                            mode='lines',
                            name='ρ₁(OLS)',
                            line=dict(color='#d32f2f', width=2, dash='dash')
                        ))
                        fig_rho.add_hline(y=1, line_dash="dot", line_color="gray",
                                          annotation_text="Unit Root")
                        fig_rho.update_layout(
                            title="Persistence Parameter Across Quantiles",
                            xaxis_title="Quantile",
                            yaxis_title="ρ₁",
                            template="plotly_white",
                            height=400,
                            legend=dict(x=0.02, y=0.98)
                        )
                        st.plotly_chart(fig_rho, use_container_width=True)

                    with col2:
                        # α₀(τ) across quantiles
                        fig_alpha = go.Figure()
                        fig_alpha.add_trace(go.Scatter(
                            x=results_df.index,
                            y=results_df['α₀(τ)'],
                            mode='lines+markers',
                            name='α₀(τ)',
                            line=dict(color='#388e3c', width=3),
                            marker=dict(size=8),
                            fill='tozeroy',
                            fillcolor='rgba(56, 142, 60, 0.1)'
                        ))
                        fig_alpha.add_trace(go.Scatter(
                            x=results_df.index,
                            y=[results_df['α₀(τ)'].mean()] * len(results_df),
                            mode='lines',
                            name='Mean α₀(τ)',
                            line=dict(color='#f57c00', width=2, dash='dash')
                        ))
                        fig_alpha.update_layout(
                            title="Intercept Across Quantiles",
                            xaxis_title="Quantile",
                            yaxis_title="α₀",
                            template="plotly_white",
                            height=400,
                            legend=dict(x=0.02, y=0.98)
                        )
                        st.plotly_chart(fig_alpha, use_container_width=True)

                    # Test statistics and critical values
                    st.subheader("📊 Test Statistics & Critical Values")

                    fig_test = go.Figure()

                    # Add t-statistic
                    fig_test.add_trace(go.Scatter(
                        x=results_df.index,
                        y=results_df['tₙ(τ)'],
                        mode='lines+markers',
                        name='tₙ(τ)',
                        line=dict(color='#1976d2', width=3),
                        marker=dict(size=10)
                    ))

                    # Add critical values
                    fig_test.add_trace(go.Scatter(
                        x=results_df.index,
                        y=results_df['CV1%'],
                        mode='lines',
                        name='CV 1%',
                        line=dict(color='#d32f2f', width=2, dash='dash')
                    ))

                    fig_test.add_trace(go.Scatter(
                        x=results_df.index,
                        y=results_df['CV5%'],
                        mode='lines',
                        name='CV 5%',
                        line=dict(color='#f57c00', width=2, dash='dash')
                    ))

                    fig_test.add_trace(go.Scatter(
                        x=results_df.index,
                        y=results_df['CV10%'],
                        mode='lines',
                        name='CV 10%',
                        line=dict(color='#fbc02d', width=2, dash='dash')
                    ))

                    fig_test.update_layout(
                        title="Test Statistics vs Critical Values",
                        xaxis_title="Quantile",
                        yaxis_title="Value",
                        template="plotly_white",
                        height=450,
                        legend=dict(x=0.02, y=0.02)
                    )
                    st.plotly_chart(fig_test, use_container_width=True)

                    # Half-lives visualization
                    st.subheader("⏱️ Half-Lives Analysis")

                    # Convert half-lives to numeric (handling '∞')
                    half_lives_numeric = []
                    for hl in results_df['Half-lives']:
                        if hl == '∞' or hl == np.inf:
                            half_lives_numeric.append(None)
                        else:
                            half_lives_numeric.append(float(hl))

                    fig_hl = go.Figure()
                    fig_hl.add_trace(go.Bar(
                        x=results_df.index,
                        y=half_lives_numeric,
                        name='Half-lives',
                        marker=dict(
                            color=half_lives_numeric,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Periods")
                        )
                    ))
                    fig_hl.update_layout(
                        title="Half-Lives Across Quantiles",
                        xaxis_title="Quantile",
                        yaxis_title="Periods",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_hl, use_container_width=True)

                with tab3:
                    st.header("Statistical Tests")

                    # Stationarity assessment
                    st.subheader("🎯 Stationarity Assessment")

                    # Count rejections at different significance levels
                    reject_1 = (results_df['tₙ(τ)'] < results_df['CV1%']).sum()
                    reject_5 = (results_df['tₙ(τ)'] < results_df['CV5%']).sum()
                    reject_10 = (results_df['tₙ(τ)'] < results_df['CV10%']).sum()
                    total = len(results_df)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        pct_1 = (reject_1 / total) * 100
                        st.metric(
                            "Reject at 1%",
                            f"{reject_1}/{total}",
                            f"{pct_1:.1f}%",
                            delta_color="normal"
                        )

                    with col2:
                        pct_5 = (reject_5 / total) * 100
                        st.metric(
                            "Reject at 5%",
                            f"{reject_5}/{total}",
                            f"{pct_5:.1f}%",
                            delta_color="normal"
                        )

                    with col3:
                        pct_10 = (reject_10 / total) * 100
                        st.metric(
                            "Reject at 10%",
                            f"{reject_10}/{total}",
                            f"{pct_10:.1f}%",
                            delta_color="normal"
                        )

                    # Pie chart for rejections
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Reject at 1%', 'Reject at 5%', 'Reject at 10%', 'Do Not Reject'],
                        values=[
                            reject_1,
                            reject_5 - reject_1,
                            reject_10 - reject_5,
                            total - reject_10
                        ],
                        marker=dict(colors=['#4caf50', '#ffc107', '#ff9800', '#e0e0e0']),
                        hole=0.4
                    )])

                    fig_pie.update_layout(
                        title="Distribution of Test Rejections",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # δ² analysis
                    st.subheader("📐 Nuisance Parameter (δ²) Analysis")

                    fig_delta = go.Figure()
                    fig_delta.add_trace(go.Scatter(
                        x=results_df.index,
                        y=results_df['δ²'],
                        mode='lines+markers',
                        name='δ²',
                        line=dict(color='#9c27b0', width=3),
                        marker=dict(size=8),
                        fill='tozeroy',
                        fillcolor='rgba(156, 39, 176, 0.1)'
                    ))

                    fig_delta.update_layout(
                        title="δ² Across Quantiles",
                        xaxis_title="Quantile",
                        yaxis_title="δ²",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_delta, use_container_width=True)

                    # Summary statistics
                    st.subheader("📊 Summary Statistics")

                    summary_stats = results_df[['ρ₁(τ)', 'α₀(τ)', 'δ²', 'tₙ(τ)']].describe()
                    st.dataframe(summary_stats.T, use_container_width=True)

                with tab4:
                    st.header("Detailed Analysis")

                    # Select specific quantile for detailed view
                    selected_q = st.selectbox(
                        "Select Quantile for Detailed Analysis",
                        options=results_df.index.tolist(),
                        format_func=lambda x: f"τ = {x:.2f}"
                    )

                    selected_row = results_df.loc[selected_q]

                    # Display detailed results for selected quantile
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        ### Quantile τ = {selected_q:.2f}

                        **Model Parameters:**
                        - Lags: {selected_row['Lags']}
                        - α₀(τ): {selected_row['α₀(τ)']:.4f}
                        - ρ₁(τ): {selected_row['ρ₁(τ)']:.4f}
                        - ρ₁(OLS): {selected_row['ρ₁(OLS)']:.4f}

                        **Test Statistics:**
                        - δ²: {selected_row['δ²']:.4f}
                        - tₙ(τ): {selected_row['tₙ(τ)']:.4f}
                        - Half-life: {selected_row['Half-lives']}
                        """)

                    with col2:
                        st.markdown(f"""
                        ### Critical Values

                        - **1% level:** {selected_row['CV1%']:.4f}
                        - **5% level:** {selected_row['CV5%']:.4f}
                        - **10% level:** {selected_row['CV10%']:.4f}

                        ### Decision
                        """)

                        if selected_row['tₙ(τ)'] < selected_row['CV1%']:
                            st.success("✅ Reject null hypothesis at 1% significance level")
                            st.info("The series is stationary at this quantile")
                        elif selected_row['tₙ(τ)'] < selected_row['CV5%']:
                            st.success("✅ Reject null hypothesis at 5% significance level")
                            st.info("The series is stationary at this quantile")
                        elif selected_row['tₙ(τ)'] < selected_row['CV10%']:
                            st.warning("⚠️ Reject null hypothesis at 10% significance level")
                            st.info("Weak evidence of stationarity at this quantile")
                        else:
                            st.error("❌ Cannot reject null hypothesis")
                            st.info("The series appears non-stationary at this quantile")

                    # Convergence warnings
                    st.markdown("---")
                    st.subheader("⚠️ Convergence Diagnostics")

                    convergence_df = pd.DataFrame(qadf.convergence_failures)
                    if not convergence_df.empty:
                        convergence_issues = convergence_df[convergence_df['convergence_failure'] == True]
                        if not convergence_issues.empty:
                            st.warning(f"⚠️ Convergence issues detected for {len(convergence_issues)} quantile(s)")
                            st.dataframe(convergence_issues, use_container_width=True)
                        else:
                            st.success("✅ All quantile regressions converged successfully")

                    # Residual analysis for selected quantile
                    st.markdown("---")
                    st.subheader("📉 Residual Analysis")

                    # Fit for the selected quantile
                    qadf_single = QADF(endog=y, model=model_type, pmax=pmax, ic=ic)
                    qadf_single.fit(selected_q)

                    residuals = qadf_single.regression.resid

                    col1, col2 = st.columns(2)

                    with col1:
                        # Residual plot
                        fig_resid = go.Figure()
                        fig_resid.add_trace(go.Scatter(
                            y=residuals,
                            mode='markers',
                            marker=dict(color='#1976d2', size=5, opacity=0.6),
                            name='Residuals'
                        ))
                        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_resid.update_layout(
                            title="Residual Plot",
                            xaxis_title="Observation",
                            yaxis_title="Residual",
                            template="plotly_white",
                            height=350
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)

                    with col2:
                        # Residual histogram
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=residuals,
                            nbinsx=30,
                            marker=dict(color='#388e3c'),
                            name='Residuals'
                        ))
                        fig_hist.update_layout(
                            title="Residual Distribution",
                            xaxis_title="Residual",
                            yaxis_title="Frequency",
                            template="plotly_white",
                            height=350,
                            showlegend=False
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                with tab5:
                    st.header("Export Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📥 Download Results")

                        # Prepare Excel file
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            results_df.to_excel(writer, sheet_name='QADF Results')

                            # Add summary sheet
                            summary_data = {
                                'Metric': ['Variable', 'Model', 'Max Lags', 'IC', 'Quantiles Tested', 'QKS Statistic'],
                                'Value': [
                                    selected_variable,
                                    model_type,
                                    pmax,
                                    ic,
                                    len(quantiles),
                                    results_df['QKS'].iloc[0]
                                ]
                            }
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                        excel_data = output.getvalue()

                        st.download_button(
                            label="📊 Download Excel Report",
                            data=excel_data,
                            file_name=f"QADF_Results_{selected_variable}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                        # CSV download
                        csv = results_df.to_csv(index=True)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"QADF_Results_{selected_variable}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:
                        st.subheader("📋 Copy Results")

                        # Format for copying
                        copy_text = results_df.to_string()
                        st.text_area(
                            "Results (copy to clipboard)",
                            value=copy_text,
                            height=300
                        )

                    # Interpretation guide
                    st.markdown("---")
                    st.subheader("📖 Interpretation Guide")

                    with st.expander("How to Interpret QADF Results", expanded=False):
                        st.markdown("""
                        ### Key Concepts:

                        **1. Quantile (τ):**
                        - Represents different parts of the distribution
                        - τ = 0.1 → Lower tail (extreme lows)
                        - τ = 0.5 → Median
                        - τ = 0.9 → Upper tail (extreme highs)

                        **2. ρ₁(τ) - Persistence Parameter:**
                        - Measures the speed of mean reversion
                        - ρ₁ = 1 → Unit root (non-stationary)
                        - ρ₁ < 1 → Stationary (mean-reverting)
                        - Lower values indicate faster mean reversion

                        **3. tₙ(τ) - Test Statistic:**
                        - Compare with critical values
                        - If tₙ(τ) < CV, reject null hypothesis of unit root
                        - Smaller values provide stronger evidence of stationarity

                        **4. Half-lives:**
                        - Time required for half the shock to dissipate
                        - Shorter half-lives indicate faster mean reversion
                        - ∞ indicates no mean reversion

                        **5. QKS Statistic:**
                        - Quantile Kolmogorov-Smirnov statistic
                        - Tests whether unit root exists at any quantile
                        - Higher values suggest rejection across quantiles

                        ### Decision Rule:
                        - **Reject H₀:** Series is stationary at that quantile
                        - **Fail to Reject:** Series has unit root at that quantile

                        ### Practical Implications:
                        - Different quantiles may show different stationarity properties
                        - Asymmetric behavior in tails vs. center
                        - Important for risk management and forecasting
                        """)
        else:
            st.warning("⚠️ No numeric columns found in the uploaded file.")

    # ---FIX WAS ADDED HERE---
    # This 'except' block closes the 'try' block from line 102 and handles errors
    # during file reading or initial processing.
    except Exception as e:
        st.error(f"❌ An error occurred while processing the uploaded file: {e}")

else:
    # Instructions when no file is uploaded
    st.info("""
    ### 👋 Welcome to the Quantile Unit Root Test Application

    **To get started:**
    1. Upload your Excel file using the sidebar
    2. Select the time series variable
    3. Configure model parameters
    4. Choose quantiles to analyze
    5. Click "Run Analysis"

    **Features:**
    - 📊 Comprehensive QADF testing across quantiles
    - 📈 Interactive visualizations
    - 🎯 Statistical significance testing
    - 💾 Export results to Excel/CSV
    - 📖 Detailed interpretation guide

    **Sample Data Format:**
    Your Excel file should contain time series data with numeric columns.
    """)

    # Show example data structure
    st.markdown("### Example Data Structure:")
    example_df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=10),
        'Variable1': np.random.randn(10).cumsum(),
        'Variable2': np.random.randn(10).cumsum()
    })
    st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Quantile Unit Root Test</strong> | Based on Koenker & Xiao (2004)</p>
    <p>For questions and support, consult the original paper or methodology documentation</p>
</div>
""", unsafe_allow_html=True)