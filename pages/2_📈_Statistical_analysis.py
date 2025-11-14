import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Statistical Analysis", layout="wide", page_icon="📈")

st.title("📈 Statistical Analysis")
st.markdown("Perform statistical tests and explore feature relationships")

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

# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Univariate Analysis", 
    "🔗 Correlation Analysis", 
    "📉 Group Comparison",
    "🧪 Hypothesis Testing"
])

with tab1:
    st.subheader("Univariate Statistical Analysis")
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for analysis")
    else:
        # Select feature for analysis
        selected_feature = st.selectbox(
            "Select a feature to analyze:",
            numeric_cols,
            key="univariate_feature"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Distribution Statistics")
            
            feature_data = data[selected_feature].dropna()
            
            # Basic statistics
            stats_dict = {
                'Count': len(feature_data),
                'Mean': feature_data.mean(),
                'Median': feature_data.median(),
                'Std Dev': feature_data.std(),
                'Min': feature_data.min(),
                'Max': feature_data.max(),
                'Q1 (25%)': feature_data.quantile(0.25),
                'Q3 (75%)': feature_data.quantile(0.75),
                'IQR': feature_data.quantile(0.75) - feature_data.quantile(0.25),
                'Skewness': stats.skew(feature_data),
                'Kurtosis': stats.kurtosis(feature_data)
            }
            
            stats_df = pd.DataFrame(
                list(stats_dict.items()),
                columns=['Statistic', 'Value']
            )
            stats_df['Value'] = stats_df['Value'].round(4)
            st.dataframe(stats_df, width='stretch', hide_index=True)
            
            # Normality test
            st.markdown("### 🔬 Normality Tests")
            if len(feature_data) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(feature_data[:5000])  # Limit for large datasets
                
                norm_tests = pd.DataFrame({
                    'Test': ['Shapiro-Wilk'],
                    'Statistic': [round(shapiro_stat, 4)],
                    'p-value': [round(shapiro_p, 4)],
                    'Normal?': ['Yes' if shapiro_p > 0.05 else 'No']
                })
                st.dataframe(norm_tests, width='stretch', hide_index=True)
                
                if shapiro_p > 0.05:
                    st.success("✅ Data appears normally distributed (p > 0.05)")
                else:
                    st.warning("⚠️ Data may not be normally distributed (p ≤ 0.05)")
        
        with col2:
            st.markdown("### 📈 Distribution Visualization")
            
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))
            
            # Histogram with KDE
            axes[0].hist(feature_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            feature_data.plot(kind='kde', ax=axes[0], color='red', linewidth=2)
            axes[0].set_xlabel(selected_feature)
            axes[0].set_ylabel('Density')
            axes[0].set_title('Distribution with KDE')
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(feature_data, dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot (Normal Distribution)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

with tab2:
    st.subheader("Correlation Analysis")
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
    else:
        # Correlation method selection
        corr_method = st.selectbox(
            "Select correlation method:",
            ["Pearson", "Spearman", "Kendall"],
            help="Pearson: linear relationships | Spearman: monotonic relationships | Kendall: ordinal associations"
        )
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr(method=corr_method.lower())
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### 🔥 {corr_method} Correlation Heatmap")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr_matrix,
                annot=False,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax,
                vmin=-1,  # Set correlation range
                vmax=1
            )
            plt.title(f'{corr_method} Correlation Matrix', fontsize=16, pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### 📊 Strongest Correlations")
            
            # Get top correlations (excluding diagonal)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
            corr_df = corr_df.sort_values('Abs Correlation', ascending=False).head(10)
            corr_df['Correlation'] = corr_df['Correlation'].round(3)
            
            st.dataframe(
                corr_df[['Feature 1', 'Feature 2', 'Correlation']],
                width='stretch',
                hide_index=True
            )
with tab3:
    st.subheader("Group Comparison Analysis")
    
    if len(categorical_cols) == 0:
        st.warning("No categorical columns found for group comparison")
    elif len(numeric_cols) == 0:
        st.warning("No numeric columns found for comparison")
    else:
        # Use stored grouping variable if available
        default_grouping = st.session_state.get('grouping_variable', categorical_cols[0])
        if default_grouping not in categorical_cols:
            default_grouping = categorical_cols[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            grouping_var = st.selectbox(
                "Select grouping variable:",
                categorical_cols,
                index=categorical_cols.index(default_grouping) if default_grouping in categorical_cols else 0,
                key="group_var",
                help=f"Currently configured: {st.session_state.get('grouping_variable', 'None')}"
            )
            
with tab4:
    st.subheader("Hypothesis Testing - All Features")
    
    if len(categorical_cols) == 0 or len(numeric_cols) == 0:
        st.warning("Need both categorical and numeric columns for hypothesis testing")
    else:
        # Select grouping variable only
        grouping_var = st.selectbox(
            "Select grouping variable:",
            categorical_cols,
            key="test_group_var",
            help="Select the categorical variable to compare groups (e.g., 'target' for control vs treatment)"
        )
        
        # Get groups
        group_names = data[grouping_var].dropna().unique()
        
        if len(group_names) < 2:
            st.warning("Need at least 2 groups for testing")
        else:
            st.markdown(f"### 🧪 Testing All Numeric Features across **{grouping_var}**")
            st.info(f"Found {len(group_names)} groups: {', '.join(map(str, group_names))}")
            
            # Perform tests for all numeric columns
            all_results = []
            
            with st.spinner(f"Running statistical tests for {len(numeric_cols)} features..."):
                for feature in numeric_cols:
                    # Get groups data for this feature
                    groups_data = []
                    for group in group_names:
                        group_values = data[data[grouping_var] == group][feature].dropna()
                        if len(group_values) > 0:
                            groups_data.append((group, group_values))
                    
                    if len(groups_data) >= 2:
                        if len(groups_data) == 2:
                            # Two-group tests
                            group1_name, group1_data = groups_data[0]
                            group2_name, group2_data = groups_data[1]
                            
                            # T-test
                            t_stat, t_pval = stats.ttest_ind(group1_data, group2_data)
                            
                            # Mann-Whitney U test
                            u_stat, u_pval = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt((group1_data.std()**2 + group2_data.std()**2) / 2)
                            if pooled_std > 0:
                                cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
                            else:
                                cohens_d = 0
                            
                            # Mean difference
                            mean_diff = group1_data.mean() - group2_data.mean()
                            
                            all_results.append({
                                'Feature': feature,
                                'Test Type': 'Two-group',
                                f'{group1_name} Mean': round(group1_data.mean(), 4),
                                f'{group2_name} Mean': round(group2_data.mean(), 4),
                                'Mean Difference': round(mean_diff, 4),
                                't-statistic': round(t_stat, 4),
                                't-test p-value': round(t_pval, 6),
                                'Mann-Whitney U': round(u_stat, 4),
                                'Mann-Whitney p-value': round(u_pval, 6),
                                "Cohen's d": round(cohens_d, 4)
                            })
                        
                        else:
                            # Multi-group tests
                            group_values = [g[1] for g in groups_data]
                            
                            # Calculate group means
                            group_means = {str(g[0]): round(g[1].mean(), 4) for g in groups_data}
                            
                            # ANOVA
                            f_stat, anova_pval = stats.f_oneway(*group_values)
                            
                            # Kruskal-Wallis
                            h_stat, kw_pval = stats.kruskal(*group_values)
                            
                            result = {
                                'Feature': feature,
                                'Test Type': 'Multi-group',
                                'F-statistic': round(f_stat, 4),
                                'ANOVA p-value': round(anova_pval, 6),
                                'Kruskal-Wallis H': round(h_stat, 4),
                                'Kruskal-Wallis p-value': round(kw_pval, 6)
                            }
                            # Add group means
                            result.update(group_means)
                            all_results.append(result)
            
            if len(all_results) == 0:
                st.warning("No features with sufficient data for testing")
            else:
                # Convert to DataFrame
                results_df = pd.DataFrame(all_results)
                
                # Add significance markers
                if 't-test p-value' in results_df.columns:
                    results_df['t-test Significant'] = results_df['t-test p-value'].apply(
                        lambda x: '✅' if x < 0.05 else '❌'
                    )
                    results_df['Mann-Whitney Significant'] = results_df['Mann-Whitney p-value'].apply(
                        lambda x: '✅' if x < 0.05 else '❌'
                    )
                elif 'ANOVA p-value' in results_df.columns:
                    results_df['ANOVA Significant'] = results_df['ANOVA p-value'].apply(
                        lambda x: '✅' if x < 0.05 else '❌'
                    )
                    results_df['Kruskal-Wallis Significant'] = results_df['Kruskal-Wallis p-value'].apply(
                        lambda x: '✅' if x < 0.05 else '❌'
                    )
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features Tested", len(results_df))
                with col2:
                    if 't-test p-value' in results_df.columns:
                        sig_count = (results_df['t-test p-value'] < 0.05).sum()
                    else:
                        sig_count = (results_df['ANOVA p-value'] < 0.05).sum()
                    st.metric("Significant Features (p < 0.05)", sig_count)
                with col3:
                    sig_pct = (sig_count / len(results_df) * 100) if len(results_df) > 0 else 0
                    st.metric("Significance Rate", f"{sig_pct:.1f}%")
                
                st.markdown("---")
                
                # Display results with highlighting
                st.markdown("**Statistical Test Results for All Features:**")
                
                # Sort by p-value
                if 't-test p-value' in results_df.columns:
                    results_df = results_df.sort_values('t-test p-value')
                    p_col = 't-test p-value'
                else:
                    results_df = results_df.sort_values('ANOVA p-value')
                    p_col = 'ANOVA p-value'
             
                def highlight_significant(row):
                    if row[p_col] < 0.001:
                        return ['background-color: #28a745'] * len(row)  # Bold green for highly significant
                    elif row[p_col] < 0.01:
                        return ['background-color: #17a2b8'] * len(row)  # Cyan/teal for very significant
                    elif row[p_col] < 0.05:
                        return ['background-color: #add8e6'] * len(row)  # Light blue for significant
                    else:
                        return [''] * len(row)
                
                styled_df = results_df.style.apply(highlight_significant, axis=1)
                st.dataframe(styled_df, width='stretch', hide_index=True)
                
                # Add legend
                st.markdown("""
                **Color Legend:**
                - 🟢 **Bold Green** (#28a745): p < 0.001 (highly significant)
                - 🔷 **Teal/Cyan** (#17a2b8): p < 0.01 (very significant)  
                - 🔵 **Light Blue** (#add8e6): p < 0.05 (significant)
                - ⚪ **No color**: p ≥ 0.05 (not significant)
                """)
                
                # Download button
                st.markdown("---")
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"statistical_tests_{grouping_var}.csv",
                    mime="text/csv"
                )
                
                # Show top significant features
                with st.expander("🔬 View Top Significant Features"):
                    sig_features = results_df[results_df[p_col] < 0.05].head(20)
                    if len(sig_features) > 0:
                        st.markdown(f"**Top {len(sig_features)} Significant Features:**")
                        st.dataframe(sig_features, width='stretch', hide_index=True)
                    else:
                        st.info("No significant features found at α = 0.05")