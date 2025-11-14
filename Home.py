import streamlit as st

def main():
    """Main landing page for RIVER-VIS"""
    st.set_page_config(
        page_title="RIVER-VIS - Bioinformatics Visualization Tool",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 RIVER-VIS")
    st.subheader("Visualization Tool for Bioinformatics Analysis")
    
    st.markdown("""
    Welcome to **RIVER-VIS** - your comprehensive toolkit for visualizing and analyzing bioinformatics data.
    
    ---
    
    ## 🎯 What is RIVER-VIS?
    
    RIVER-VIS (Rapid Interactive Visualization Environment for Research - VISualization) is a powerful 
    web-based application designed to streamline bioinformatics analysis workflows. Built with Streamlit, 
    it provides an intuitive interface for exploring gene expression data, performing statistical analyses, 
    and generating publication-ready visualizations.
    
    ## 🚀 Key Features
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        ### 📊 Raincloud Plots
        
        Visualize features distributions with beautiful raincloud plots that combine:
        - Box plots
        - Violin plots
        - Jittered data points
        - Autometic statistical testing
        """)
    
    # with col2:
    #     st.info("""
    #     ### 📈 Statistical Analysis
        
    #     Perform comprehensive statistical tests:
    #     - T-tests
    #     - ANOVA
    #     - Non-parametric tests
    #     - Multiple comparison corrections
    #     """)
    
    # with col3:
    #     st.info("""
    #     ### 📁 Data Management
        
    #     Efficient data handling:
    #     - Upload various formats
    #     - Filter and subset data
    #     - Export results
    #     - Interactive exploration
    #     """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📖 Getting Started
    
    1. **Navigate** using the sidebar menu (👈)
    2. **Upload** your bioinformatics data
    3. **Visualize** and analyze your results
    4. **Export** publication-ready figures
    
    ### 💡 Quick Tips
    - Start with the **Raincloud Plot** page to visualize your gene expression data
    - Use the **Statistical Analysis** page for hypothesis testing
    - Explore the **Data Explorer** to understand your dataset
    
    ---
    
    *RIVER-VIS is designed for researchers, bioinformaticians, and data scientists working with 
    high-throughput sequencing data, microarray analyses, and other omics datasets.*
    """)
    
    # Optional: Add some metrics or stats
    st.markdown("### 📊 Quick Stats")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # with metric_col1:
    #     st.metric("Visualization Types", "5+")
    # with metric_col2:
    #     st.metric("Statistical Tests", "10+")
    # with metric_col3:
    #     st.metric("File Formats", "Multiple")
    # with metric_col4:
    #     st.metric("Export Options", "PDF, PNG, SVG")

if __name__ == "__main__":
    main()