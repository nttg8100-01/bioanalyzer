import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from scipy import stats as sp_stats
from plotting.raincloud import make_raincloud_plot
from plotting.fonts import get_available_fonts
from stats.tests import choose_test

st.set_page_config(page_title="Raincloud Plot", layout="wide")

def sidebar_controls(genes=None, unique_conditions=None, default_controls=None):
    """Main sidebar controls for the raincloud plot page"""
    # Data source section
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["session", "upload"])
    uploaded_file = st.sidebar.file_uploader("Upload CSV file...") if data_source == "upload" else None
    
    # Plot customization section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Visualization Options")
    
    # Plot type
    plot_type = st.sidebar.selectbox(
        "Select plot type:",
        ["Bar Plot with Points", "Raincloud Plot"],
        index=1,  # Default to Raincloud
        help="Bar plot recommended for small samples"
    )
    
    # Error bar type (only for bar plot)
    error_type = st.sidebar.radio(
        "Error bars:",
        ["SEM (Standard Error)", "SD (Standard Deviation)", "95% CI"],
        help="SEM: Standard Error of Mean | SD: Standard Deviation | CI: Confidence Interval"
    ) if plot_type == "Bar Plot with Points" else None
    
    # Show points (only for bar plot)
    show_points = st.sidebar.checkbox("Show individual points", value=True) if plot_type == "Bar Plot with Points" else None
    
    # Group labels (only show if we have conditions)
    if unique_conditions is not None and len(unique_conditions) >= 2:
        group1_label = st.sidebar.text_input("Group 1 Label", str(unique_conditions[0]).capitalize())
        group2_label = st.sidebar.text_input("Group 2 Label", str(unique_conditions[1]).capitalize())
    else:
        group1_label = "Group 1"
        group2_label = "Group 2"
    
    # Plot formatting
    if default_controls is None:
        default_controls = {}
    
    plot_title = st.sidebar.text_input("Plot Title", default_controls.get("plot_title", "Gene Expression"))
    x_label = st.sidebar.text_input("X Axis Label", default_controls.get("x_label", "Condition"))
    y_label = st.sidebar.text_input("Y Axis Label", default_controls.get("y_label", "Expression Level"))
    
    # Styling options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Style Options")
    
    # Fonts
    available_fonts, missing_fonts = get_available_fonts()
    font_family = st.sidebar.selectbox("Font Family", available_fonts)
    
    # Colors
    color1 = st.sidebar.color_picker("Group 1 Color", "#1F77B4")
    color2 = st.sidebar.color_picker("Group 2 Color", "#D62728")
    
    # Transparency settings (for raincloud plot)
    if plot_type == "Raincloud Plot":
        violin_alpha = st.sidebar.slider("Violin Transparency", 0.0, 1.0, 0.6, 0.1)
        boxplot_alpha = st.sidebar.slider("Box Transparency", 0.0, 1.0, 0.8, 0.1)
        jitter_alpha = st.sidebar.slider("Point Transparency", 0.0, 1.0, 0.5, 0.1)
        point_size = st.sidebar.slider("Point Size", 1, 5, 2)
        group_spacing = st.sidebar.slider("Group Spacing", 0.5, 2.0, 2.0, 0.1)
    else:
        violin_alpha = boxplot_alpha = jitter_alpha = point_size = group_spacing = None
    
    # Figure size
    fig_width = st.sidebar.slider("Figure Width", 4, 16, default_controls.get("fig_width", 8))
    fig_height = st.sidebar.slider("Figure Height", 4, 16, default_controls.get("fig_height", 6))
    
    # Export section
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Export Options")
    
    dpi = st.sidebar.selectbox("Resolution (DPI)", [72, 150, 300, 600], index=2)
    plot_format = st.sidebar.selectbox("File Format", ["PNG", "PDF", "SVG", "JPEG"], index=0)
    generate_plot_download = st.sidebar.button("Generate Plot Download")
    
    return {
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "plot_type": plot_type,
        "error_type": error_type,
        "show_points": show_points,
        "group1_label": group1_label,
        "group2_label": group2_label,
        "plot_title": plot_title,
        "x_label": x_label,
        "y_label": y_label,
        "font_family": font_family,
        "color1": color1,
        "color2": color2,
        "violin_alpha": violin_alpha,
        "boxplot_alpha": boxplot_alpha,
        "jitter_alpha": jitter_alpha,
        "point_size": point_size,
        "group_spacing": group_spacing,
        "fig_width": fig_width,
        "fig_height": fig_height,
        "dpi": dpi,
        "plot_format": plot_format,
        "generate_plot_download": generate_plot_download
    }

def transform_data_for_raincloud(data, gene_col, group_col):
    """Transform data from wide to long format suitable for raincloud plotting."""
    plot_data = pd.DataFrame({
        'value': data[gene_col].values,
        'condition': data[group_col].values,
        'gene': gene_col
    })
    
    # Remove NaN values
    plot_data = plot_data.dropna()
    return plot_data

def create_bar_plot(df, sidebar_vals, pval):
    """Create bar plot with error bars and significance testing"""
    means = df.groupby('condition')['expression'].mean()
    sems = df.groupby('condition')['expression'].sem()
    stds = df.groupby('condition')['expression'].std()
    counts = df.groupby('condition')['expression'].count()

    if sidebar_vals["error_type"] == "SEM (Standard Error)":
        errors = sems
    elif sidebar_vals["error_type"] == "SD (Standard Deviation)":
        errors = stds
    else:
        errors = sems * sp_stats.t.ppf(0.975, counts - 1)

    fig, ax = plt.subplots(figsize=(sidebar_vals['fig_width'], sidebar_vals['fig_height']))
    x_pos = np.arange(len(means))
    colors = [sidebar_vals['color1'], sidebar_vals['color2']]

    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, 
                 alpha=0.8, color=colors, edgecolor='black', 
                 linewidth=2, error_kw={'linewidth': 2.5, 'elinewidth': 2.5, 'capthick': 2.5})

    if sidebar_vals["show_points"]:
        for i, cond in enumerate(means.index):
            cond_data = df[df['condition'] == cond]['expression'].values
            np.random.seed(42)
            jitter = np.random.normal(0, 0.05, len(cond_data))
            x_vals = np.full(len(cond_data), i) + jitter
            ax.scatter(x_vals, cond_data, color='black', s=80, 
                      alpha=0.6, zorder=3, edgecolors='white', linewidth=1.5)

    # Add significance bars
    y_max = df['expression'].max()
    y_min = df['expression'].min()
    y_range = y_max - y_min
    max_bar_height = max([means.iloc[i] + errors.iloc[i] for i in range(len(means))])
    y_sig_line = max_bar_height + y_range * 0.05
    y_sig_text = y_sig_line + y_range * 0.03

    ax.plot([0, 1], [y_sig_line, y_sig_line], 'k-', linewidth=2)
    ax.plot([0, 0], [y_sig_line - y_range * 0.01, y_sig_line], 'k-', linewidth=2)
    ax.plot([1, 1], [y_sig_line - y_range * 0.01, y_sig_line], 'k-', linewidth=2)

    # Significance symbols
    if pval < 0.001:
        sig_symbol = "***"
        sig_text = f"p < 0.001"
    elif pval < 0.01:
        sig_symbol = "**"
        sig_text = f"p = {pval:.3f}"
    elif pval < 0.05:
        sig_symbol = "*"
        sig_text = f"p = {pval:.3f}"
    else:
        sig_symbol = "ns"
        sig_text = f"p = {pval:.3f}"

    ax.text(0.5, y_sig_text, sig_symbol, ha='center', va='bottom', fontsize=18, fontweight='bold')
    ax.text(0.5, y_sig_text + y_range * 0.08, sig_text, ha='center', va='bottom', fontsize=18, style='italic')

    # Style the plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(means.index, fontsize=13, fontweight='bold')
    ax.set_title(sidebar_vals["plot_title"], fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(sidebar_vals["y_label"], fontsize=13, fontweight='bold')
    ax.set_xlabel(sidebar_vals["x_label"], fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_ylim(bottom=min(0, y_min - y_range * 0.05), top=y_sig_text + y_range * 0.15)

    return fig

"""Main function for the raincloud plot page"""
st.title("📊 Raincloud Plot for Gene Expression")

# Check if data exists in session state
if 'data' not in st.session_state or st.session_state['data'] is None:
    st.warning("⚠️ No data loaded! Please go to the Data Loader page first.")
    st.info("👈 Navigate to **Data Loader** in the sidebar to upload or load example data.")
    
    if st.button("📁 Go to Data Loader"):
        st.switch_page("pages/1_📁_Data_Loader.py")
    
    st.stop()

# Get data from session state
data = st.session_state['data']

# Show data info
with st.expander("ℹ️ Current Dataset Info"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(data):,}")
    with col2:
        st.metric("Columns", f"{len(data.columns):,}")
    with col3:
        if st.session_state.get('using_example', False):
            st.info("📊 Using Example Data")
        else:
            st.success("✅ Custom Data Loaded")

# Data transformation section
st.markdown("---")
st.subheader("🔧 Data Selection and Transformation")

# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

if len(numeric_cols) == 0:
    st.error("❌ No numeric columns found in the data. Cannot create raincloud plot.")
    st.stop()

if len(categorical_cols) == 0:
    st.error("❌ No categorical columns found in the data. Need a grouping variable.")
    st.stop()

# Selection controls
col1, col2 = st.columns(2)

with col1:
    selected_gene = st.selectbox(
        "Select gene/feature to visualize:",
        numeric_cols,
        help="Choose the numeric column representing gene expression values"
    )

with col2:
    default_grouping = st.session_state.get('grouping_variable', categorical_cols[0])
    if default_grouping not in categorical_cols:
        default_grouping = categorical_cols[0]
    
    grouping_var = st.selectbox(
        "Select grouping variable:",
        categorical_cols,
        index=categorical_cols.index(default_grouping) if default_grouping in categorical_cols else 0,
        help=f"Configured in Data Loader: {st.session_state.get('grouping_variable', 'Not set')}"
    )

# Transform the data
try:
    transformed_data = transform_data_for_raincloud(data, selected_gene, grouping_var)
    
    # Show preview of transformed data
    with st.expander("📊 Preview Transformed Data"):
        st.markdown(f"**Data shape:** {transformed_data.shape[0]} rows × {transformed_data.shape[1]} columns")
        
        # Group summary
        group_summary = transformed_data.groupby('condition')['value'].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std', 'std')
        ]).round(4)
        
        st.markdown(f"**Summary by {grouping_var}:**")
        st.dataframe(group_summary, width='stretch')
        
        st.markdown("**First 10 rows:**")
        st.dataframe(transformed_data.head(10), width='stretch', hide_index=True)
        
        unique_groups = transformed_data['condition'].unique()
        st.info(f"**Groups found:** {', '.join(map(str, unique_groups))}")
    
    st.success(f"✅ Data transformed successfully! Ready to plot **{selected_gene}** by **{grouping_var}**")
    
except Exception as e:
    st.error(f"❌ Error transforming data: {str(e)}")
    st.exception(e)
    st.stop()

st.markdown("---")

# Get unique conditions for sidebar controls
unique_conditions = transformed_data["condition"].unique()

# Get controls from sidebar
controls = sidebar_controls(
    genes=[selected_gene],
    unique_conditions=unique_conditions,
    default_controls={"plot_title": f"{selected_gene} Expression", "x_label": grouping_var, "y_label": "Expression Level"}
)

# Plot section
if len(unique_conditions) < 2:
    st.error("Need at least 2 conditions!")

# Prepare data for plotting
df = transformed_data[["condition", "value"]].rename(columns={"value": "expression"})
group1_label = controls["group1_label"]
group2_label = controls["group2_label"]

df["condition"] = df["condition"].replace({
    unique_conditions[0]: group1_label, 
    unique_conditions[1]: group2_label
})
df = df[df["condition"].isin([group1_label, group2_label])]

group1 = df[df["condition"] == group1_label]["expression"]
group2 = df[df["condition"] == group2_label]["expression"]

if len(group1) < 3 or len(group2) < 3:
    st.error("Each group needs at least 3 samples for statistical testing!")

# Perform statistical test
test_name, stat, pval, p1, p2 = choose_test(group1, group2)

# Create plot based on selection
if controls["plot_type"] == "Bar Plot with Points":
    fig = create_bar_plot(df, controls, pval)
else:
    plot = make_raincloud_plot(df, controls, selected_gene, pval, test_name)
    fig = plot.draw()

# Display plot
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.pyplot(fig)

# Statistical analysis results
st.subheader("Statistical Analysis")
col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.metric("Test", test_name)
with col2: 
    st.metric("Statistic", f"{stat:.4f}")
with col3: 
    st.metric("P-value", f"{pval:.4f}")
with col4:
    if pval < 0.001:
        sig = "*** (p < 0.001)"
    elif pval < 0.01:
        sig = "** (p < 0.01)"
    elif pval < 0.05:
        sig = "* (p < 0.05)"
    else:
        sig = "ns (not significant)"
    st.metric("Significance", sig)

# Additional information expandables
with st.expander("📊 Normality Test Results (Shapiro-Wilk)"):
    st.write(f"{group1_label}: p = {p1:.4f} {'(Normal)' if p1 > 0.05 else '(Not normal)'}")
    st.write(f"{group2_label}: p = {p2:.4f} {'(Normal)' if p2 > 0.05 else '(Not normal)'}")
    st.write("---")
    st.write("Interpretation: If p > 0.05, data is considered normal.")

with st.expander("📈 Summary Statistics"):
    summary_stats = df.groupby("condition")["expression"].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    st.dataframe(summary_stats, width='stretch')

with st.expander("📋 View Raw Data"):
    st.dataframe(df, width='stretch', hide_index=True)

# Download options
csv_data = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "📄 Download Data (CSV)",
    data=csv_data,
    file_name=f"{selected_gene}_expression_data.csv",
    mime="text/csv"
)

# Plot download
if controls["generate_plot_download"]:
    buf = io.BytesIO()
    fig.savefig(buf, format=controls["plot_format"].lower(), dpi=controls["dpi"], bbox_inches='tight')
    buf.seek(0)
    mimes = {
        "PNG": "image/png", "PDF": "application/pdf",
        "SVG": "image/svg+xml", "JPEG": "image/jpeg"
    }
    st.sidebar.download_button(
        f"📥 Download {controls['plot_format']}",
        data=buf,
        file_name=f"{selected_gene}_{controls['plot_type'].replace(' ', '_').lower()}.{controls['plot_format'].lower()}",
        mime=mimes[controls["plot_format"]]
    )
    st.sidebar.success(f"✅ Plot ready at {controls['dpi']} DPI")
    plt.close(fig)
