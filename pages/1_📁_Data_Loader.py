import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(
    page_title="RIVER-VIS - Data Loader",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Data Loader")
st.markdown("Upload and explore your bioinformatics data")

# Add option to use example dataset
col_upload, col_example = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Choose a file (CSV, TSV, Excel)",
        type=["csv", "tsv", "txt", "xlsx"],
        help="Upload your bioinformatics dataset"
    )

with col_example:
    st.markdown("### Or try an example:")
    use_example = st.button("📊 Load Example Gene Expression Data", use_container_width=True)

# Create example dataset
@st.cache_data
def create_example_data():
    """Generate example gene expression dataset"""
    np.random.seed(42)
    n_genes = 100
    
    # Create sample names as row indices and gene names as columns
    samples = ['control_1', 'control_2', 'control_3', 'treatment_1', 'treatment_2', 'treatment_3']
    gene_names = [f'Gene_{chr(65 + i % 26)}{i}' for i in range(n_genes)]
    
    # Generate expression data matrix (samples x genes)
    expression_data = np.random.lognormal(2, 0.5, (len(samples), n_genes))
    
    # Add some differential expression for treatment samples
    expression_data[3:6, :] *= np.random.lognormal(0.2, 0.3, n_genes)
    
    df = pd.DataFrame(expression_data, index=samples, columns=gene_names)
    df["target"] = ['control']*3 + ['treatment']*3
    return df

# Handle example data loading
if use_example:
    st.session_state['using_example'] = True
    st.session_state['data'] = create_example_data()
    uploaded_file = None

# Process data
if uploaded_file is not None or st.session_state.get('using_example', False):
    try:
        # Load data with pandas
        if uploaded_file is not None:
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
            
            with st.spinner("Loading data..."):
                # Read based on file extension
                if uploaded_file.name.endswith('.csv'):
                    pdf = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.tsv', '.txt')):
                    pdf = pd.read_csv(uploaded_file, sep='\t')
                elif uploaded_file.name.endswith('.xlsx'):
                    pdf = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    st.stop()
            
            st.session_state['using_example'] = False
            
        else:
            # Use example data
            st.info("📊 Using example gene expression dataset")
            pdf = st.session_state['data']
        
        # Store in session state for use in other pages
        st.session_state['data'] = pdf
        
        # === NEW: Grouping Variable Selection ===
        st.markdown("---")
        st.subheader("🎯 Configure Grouping Variable")
        
        # Identify categorical columns
        categorical_cols = pdf.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) > 0:
            col_select, col_info = st.columns([2, 1])
            
            with col_select:
                # Get previously selected grouping variable if exists
                default_group = st.session_state.get('grouping_variable', categorical_cols[0])
                if default_group not in categorical_cols:
                    default_group = categorical_cols[0]
                
                grouping_var = st.selectbox(
                    "Select the grouping/condition variable:",
                    categorical_cols,
                    index=categorical_cols.index(default_group) if default_group in categorical_cols else 0,
                    help="This column will be used for group comparisons in statistical analysis and visualizations (e.g., 'target' for control vs treatment)"
                )
                
                # Store in session state
                st.session_state['grouping_variable'] = grouping_var
            
            with col_info:
                # Show group information
                unique_groups = pdf[grouping_var].dropna().unique()
                st.info(f"**Groups found:** {len(unique_groups)}")
                st.write(f"**Labels:** {', '.join(map(str, unique_groups))}")
            
            # Display group summary
            with st.expander("📊 View Group Distribution"):
                group_counts = pdf[grouping_var].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Group Counts:**")
                    count_df = pd.DataFrame({
                        'Group': group_counts.index.astype(str),
                        'Count': group_counts.values,
                        'Percentage': (group_counts.values / len(pdf) * 100).round(2)
                    })
                    st.dataframe(count_df, width='stretch', hide_index=True)
                
                with col2:
                    st.markdown("**Distribution:**")
                    st.bar_chart(group_counts)
            
            st.success(f"✅ Grouping variable set to: **{grouping_var}**")
        else:
            st.warning("⚠️ No categorical columns found. Add a grouping variable for statistical analysis.")
            st.session_state['grouping_variable'] = None
        
        # === END NEW SECTION ===
        
        # Display tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📊 Data Types", "📈 Statistics", "🔍 Quality"])
        
        with tab1:
            st.subheader("Data Preview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(pdf):,}")
            with col2:
                st.metric("Total Columns", f"{len(pdf.columns):,}")
            with col3:
                st.metric("Memory Usage", f"{pdf.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show first/last rows
            st.markdown("**First 10 rows:**")
            st.dataframe(pdf.head(10), width='stretch')
            
            with st.expander("Show last 10 rows"):
                st.dataframe(pdf.tail(10), width='stretch')
        
        with tab2:
            st.subheader("Data Types")
            
            # Get data types - convert to native Python types
            dtypes_df = pd.DataFrame({
                'Column': [str(col) for col in pdf.dtypes.index],
                'Data Type': [str(dtype) for dtype in pdf.dtypes.values],
                'Non-Null Count': [int(count) for count in pdf.count().values],
                'Null Count': [int(null_count) for null_count in pdf.isnull().sum().values],
                'Null %': [float(pct) for pct in (pdf.isnull().sum() / len(pdf) * 100).round(2).values]
            })
            
            st.dataframe(dtypes_df, width='stretch')
        
        with tab3:
            st.subheader("Descriptive Statistics")
            
            # Numeric columns statistics
            numeric_cols = pdf.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Columns:**")
                desc_stats = pdf[numeric_cols].describe()
                
                # Add missing data info
                missing_counts = pdf[numeric_cols].isnull().sum()
                missing_pct = (missing_counts / len(pdf) * 100).round(2)
                
                # Create new rows for the stats table
                desc_stats.loc['missing'] = missing_counts
                desc_stats.loc['missing %'] = missing_pct
                
                # Display transposed (columns as rows)
                st.dataframe(desc_stats.T.reset_index().rename(columns={'index': 'Column'}), 
                            width='stretch', hide_index=True)
            else:
                st.info("No numeric columns found")
        
        with tab4:
            st.subheader("Data Quality Summary")
            
            # Overall quality metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Completeness:**")
                quality_metrics = {
                    'Total Cells': len(pdf) * len(pdf.columns),
                    'Missing Cells': int(pdf.isnull().sum().sum()),
                    'Complete Rows': len(pdf.dropna()),
                    'Completeness %': round((1 - pdf.isnull().sum().sum() / (len(pdf) * len(pdf.columns))) * 100, 2)
                }
                
                for metric, value in quality_metrics.items():
                    st.metric(metric, value)
            
            with col2:
                st.markdown("**Uniqueness:**")
                duplicate_metrics = {
                    'Duplicate Rows': int(pdf.duplicated().sum()),
                    'Unique Rows': len(pdf) - int(pdf.duplicated().sum()),
                    'Duplicate %': round((pdf.duplicated().sum() / len(pdf)) * 100, 2)
                }
                
                for metric, value in duplicate_metrics.items():
                    st.metric(metric, value)
            
            # Missing data heatmap per column
            st.markdown("**Missing Data by Column:**")
            missing_by_col = pdf.isnull().sum().sort_values(ascending=False)
            missing_pct = (missing_by_col / len(pdf) * 100).round(2)
            
            missing_df = pd.DataFrame({
                'Column': [str(col) for col in missing_by_col.index],
                'Missing Count': [int(x) for x in missing_by_col.values],
                'Missing %': [float(x) for x in missing_pct.values]
            })
            
            st.dataframe(missing_df, width='stretch')
            st.bar_chart(missing_df.set_index('Column')['Missing %'])
        
        # Download processed data
        st.markdown("---")
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = pdf.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Convert to Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                pdf.to_excel(writer, index=False, sheet_name='Data')
            
            st.download_button(
                label="📊 Download as Excel",
                data=buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            json_str = pdf.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="📋 Download as JSON",
                data=json_str,
                file_name="processed_data.json",
                mime="application/json"
            )
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload a file or try the example dataset to get started")
    
    # Show example of expected data format
    with st.expander("📖 Expected Data Format"):
        st.markdown("""
        ### Supported File Formats:
        - **CSV**: Comma-separated values
        - **TSV/TXT**: Tab-separated values
        - **Excel**: .xlsx files
        
        ### Example Gene Expression Data Structure:
        ```
        sample,target,Gene_A0,Gene_B1,Gene_C2,...
        control_1,control,12.5,13.2,15.8,...
        control_2,control,12.3,13.0,15.6,...
        treatment_1,treatment,15.8,16.1,18.5,...
        treatment_2,treatment,15.6,16.0,18.3,...
        ```
        
        The example dataset includes:
        - 100 genes with expression values (columns)
        - Control and treatment groups (3 replicates each)
        - A 'target' column for grouping (control vs treatment)
        """)