dtypes_df = pd.DataFrame({
'Column': [str(col) for col in pdf.dtypes.index],
'Data Type': [str(dtype) for dtype in pdf.dtypes.values],
'Non-Null Count': [int(count) for count in pdf.count().values],
'Null Count': [int(null_count) for null_count in pdf.isnull().sum().values],
'Null %': [float(pct) for pct in (pdf.isnull().sum() / len(pdf) * 100).round(2).values]
})