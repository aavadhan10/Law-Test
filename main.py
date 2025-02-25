import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import numpy as np

st.set_page_config(layout="wide", page_title="Law Firm Analytics Dashboard")

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def convert_df_dtypes(df):
    """Convert DataFrame column types appropriately."""
    for col in df.columns:
        # Convert any columns that mention 'date' to datetime
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        # Convert numeric columns
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    return df

def format_column_name(col):
    """Make column names more readable."""
    return col.replace('_', ' ').title()

# Main app header
st.title("Law Firm Analytics Dashboard")
st.markdown("### Upload your Clio data export to generate insights")

# Create sidebar for filters and settings
st.sidebar.title("Dashboard Controls")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Clean and convert data types
    df = convert_df_dtypes(df)
    
    # Ask about filters
    st.sidebar.header("Filter Options")
    st.sidebar.info("Select the filters you want to include in your dashboard")
    
    # Create filter options based on available columns
    filter_options = {}
    
    # Date range filter if date columns exist
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        show_date_filter = st.sidebar.checkbox("Date Range Filter", value=True)
        if show_date_filter:
            selected_date_col = st.sidebar.selectbox("Select date column for filtering", date_cols)
            try:
                min_date = df[selected_date_col].min()
                max_date = df[selected_date_col].max()
                date_range = st.sidebar.date_input(
                    "Select date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filter_options['date_range'] = {
                        'column': selected_date_col,
                        'start': start_date,
                        'end': end_date
                    }
                    df = df[(df[selected_date_col] >= pd.Timestamp(start_date)) & 
                            (df[selected_date_col] <= pd.Timestamp(end_date))]
            except:
                st.sidebar.warning(f"Unable to filter by {selected_date_col}. Check the data format.")
    
    # Attorney filter
    if 'User full name (first, last)' in df.columns:
        show_attorney_filter = st.sidebar.checkbox("Attorney Filter", value=True)
        if show_attorney_filter:
            attorneys = ['All'] + sorted(df['User full name (first, last)'].unique().tolist())
            selected_attorneys = st.sidebar.multiselect("Select Attorneys", attorneys, default=['All'])
            if 'All' not in selected_attorneys:
                df = df[df['User full name (first, last)'].isin(selected_attorneys)]
                filter_options['attorneys'] = selected_attorneys
    
    # Practice area filter
    if 'Practice area' in df.columns:
        show_practice_filter = st.sidebar.checkbox("Practice Area Filter", value=True)
        if show_practice_filter:
            practice_areas = ['All'] + sorted(df['Practice area'].unique().tolist())
            selected_practice = st.sidebar.multiselect("Select Practice Areas", practice_areas, default=['All'])
            if 'All' not in selected_practice:
                df = df[df['Practice area'].isin(selected_practice)]
                filter_options['practice_areas'] = selected_practice
    
    # Matter status filter
    if 'Matter status' in df.columns:
        show_status_filter = st.sidebar.checkbox("Matter Status Filter", value=True)
        if show_status_filter:
            statuses = ['All'] + sorted(df['Matter status'].unique().tolist())
            selected_status = st.sidebar.multiselect("Select Matter Status", statuses, default=['All'])
            if 'All' not in selected_status:
                df = df[df['Matter status'].isin(selected_status)]
                filter_options['matter_status'] = selected_status
    
    # Billable filter
    if 'Billable matter' in df.columns:
        show_billable_filter = st.sidebar.checkbox("Billable Matters Only", value=False)
        if show_billable_filter:
            df = df[df['Billable matter'] == 1]
            filter_options['billable_only'] = True
    
    # Custom numeric range filter
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    if numeric_cols:
        show_numeric_filter = st.sidebar.checkbox("Numeric Range Filter")
        if show_numeric_filter:
            numeric_col = st.sidebar.selectbox("Select numeric column for filtering", numeric_cols)
            min_val = float(df[numeric_col].min())
            max_val = float(df[numeric_col].max())
            range_filter = st.sidebar.slider(
                f"Filter by {numeric_col} range",
                min_val,
                max_val,
                (min_val, max_val)
            )
            df = df[(df[numeric_col] >= range_filter[0]) & (df[numeric_col] <= range_filter[1])]
            filter_options['numeric_range'] = {
                'column': numeric_col,
                'range': range_filter
            }
    
    # Show active filters
    if filter_options:
        st.sidebar.subheader("Active Filters")
        for filter_type, filter_value in filter_options.items():
            if isinstance(filter_value, dict):
                if filter_type == 'date_range':
                    st.sidebar.write(f"📅 Date: {filter_value['start']} to {filter_value['end']}")
                elif filter_type == 'numeric_range':
                    st.sidebar.write(f"🔢 {filter_value['column']}: {filter_value['range'][0]} to {filter_value['range'][1]}")
            elif isinstance(filter_value, list):
                st.sidebar.write(f"🔍 {filter_type.title()}: {', '.join(filter_value)}")
            else:
                st.sidebar.write(f"🔍 {filter_type.title()}: {filter_value}")
    
    # Display data overview
    st.subheader("Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total records:** {len(df)}")
    with col2:
        st.write(f"**Total columns:** {len(df.columns)}")
    
    # Allow users to view the raw data if they want
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    # Display column information
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Values': df.count().values,
            'Null Values': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)
    
    # Option to add more data
    add_more_data = st.checkbox("Would you like to add more data?")
    
    if add_more_data:
        st.info("This feature would allow connecting to additional data sources or uploading supplementary files.")
        # This would be implemented based on specific requirements
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Default Visualizations", "Custom Analysis", "Attorney Performance", "Matter Insights", "Real-Time Connection"])
    
    with tab1:
        st.header("Default Visualizations")
        
        # Check for required columns
        required_cols = ['Utilization rate', 'Billed hours', 'Unbilled hours', 'User full name (first, last)', 'Practice area']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Some expected columns are missing: {', '.join(missing_cols)}")
            st.info("The default visualizations may not all be available.")
        
        # Create default visualizations based on available columns
        default_viz_cols = st.columns(2)
        
        with default_viz_cols[0]:
            # Visualization 1: Utilization Rate by Attorney
            if 'Utilization rate' in df.columns and 'User full name (first, last)' in df.columns:
                util_by_atty = df.groupby('User full name (first, last)')['Utilization rate'].mean().reset_index()
                util_by_atty = util_by_atty.sort_values('Utilization rate', ascending=False)
                
                fig = px.bar(
                    util_by_atty, 
                    x='User full name (first, last)', 
                    y='Utilization rate',
                    title='Average Utilization Rate by Attorney',
                    labels={'User full name (first, last)': 'Attorney', 'Utilization rate': 'Utilization Rate (%)'},
                    color='Utilization rate',
                    color_continuous_scale='blues'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with default_viz_cols[1]:
            # Visualization 2: Billed vs Unbilled Hours
            if 'Billed hours' in df.columns and 'Unbilled hours' in df.columns:
                billed_vs_unbilled = pd.DataFrame({
                    'Category': ['Billed Hours', 'Unbilled Hours'],
                    'Hours': [df['Billed hours'].sum(), df['Unbilled hours'].sum()]
                })
                
                fig = px.pie(
                    billed_vs_unbilled,
                    values='Hours',
                    names='Category',
                    title='Billed vs Unbilled Hours',
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig, use_container_width=True)
        
        default_viz_cols2 = st.columns(2)
        
        with default_viz_cols2[0]:
            # Visualization 3: Practice Area Distribution
            if 'Practice area' in df.columns:
                practice_dist = df['Practice area'].value_counts().reset_index()
                practice_dist.columns = ['Practice Area', 'Count']
                
                fig = px.bar(
                    practice_dist,
                    x='Practice Area',
                    y='Count',
                    title='Matter Distribution by Practice Area',
                    color='Count',
                    color_continuous_scale='blues'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with default_viz_cols2[1]:
            # Visualization 4: Monthly Trend of Billed Hours
            if 'Billed hours' in df.columns and 'Activity month' in df.columns:
                monthly_billed = df.groupby('Activity month')['Billed hours'].sum().reset_index()
                
                fig = px.line(
                    monthly_billed,
                    x='Activity month',
                    y='Billed hours',
                    title='Monthly Trend of Billed Hours',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Custom Analysis")
        
        # Allow users to select dimensions for custom visualization
        st.subheader("Create Your Own Visualization")
        
        viz_type = st.selectbox(
            "Select Visualization Type", 
            ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Heatmap"]
        )
        
        numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if viz_type in ["Bar Chart", "Line Chart"]:
            x_axis = st.selectbox("Select X-Axis (Category)", categorical_cols)
            y_axis = st.selectbox("Select Y-Axis (Value)", numeric_cols)
            agg_func = st.selectbox("Select Aggregation Function", ["Sum", "Average", "Count", "Median", "Min", "Max"])
            
            # Map the aggregation function to pandas function
            agg_map = {
                "Sum": "sum",
                "Average": "mean",
                "Count": "count",
                "Median": "median",
                "Min": "min",
                "Max": "max"
            }
            
            if st.button("Generate Visualization"):
                # Group by the selected category and apply the aggregation function
                grouped_data = df.groupby(x_axis)[y_axis].agg(agg_map[agg_func]).reset_index()
                
                if viz_type == "Bar Chart":
                    fig = px.bar(
                        grouped_data, 
                        x=x_axis, 
                        y=y_axis,
                        title=f"{agg_func} of {y_axis} by {x_axis}",
                        color=y_axis,
                        color_continuous_scale='blues'
                    )
                else:  # Line Chart
                    fig = px.line(
                        grouped_data, 
                        x=x_axis, 
                        y=y_axis,
                        title=f"{agg_func} of {y_axis} by {x_axis}",
                        markers=True
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart":
            category = st.selectbox("Select Category for Segments", categorical_cols)
            value = st.selectbox("Select Value for Size", numeric_cols)
            
            if st.button("Generate Visualization"):
                # Group by the selected category and sum the values
                pie_data = df.groupby(category)[value].sum().reset_index()
                
                fig = px.pie(
                    pie_data,
                    values=value,
                    names=category,
                    title=f"Distribution of {value} by {category}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            x_value = st.selectbox("Select X-Axis Value", numeric_cols)
            y_value = st.selectbox("Select Y-Axis Value", numeric_cols)
            color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            
            if st.button("Generate Visualization"):
                if color_by == "None":
                    fig = px.scatter(
                        df,
                        x=x_value,
                        y=y_value,
                        title=f"Scatter Plot: {y_value} vs {x_value}"
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_value,
                        y=y_value,
                        color=color_by,
                        title=f"Scatter Plot: {y_value} vs {x_value}, colored by {color_by}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Heatmap":
            row_cat = st.selectbox("Select Row Category", categorical_cols)
            col_cat = st.selectbox("Select Column Category", categorical_cols)
            value_col = st.selectbox("Select Value", numeric_cols)
            
            if st.button("Generate Visualization"):
                # Create a pivot table for the heatmap
                heatmap_data = df.pivot_table(
                    index=row_cat,
                    columns=col_cat,
                    values=value_col,
                    aggfunc='mean'
                ).fillna(0)
                
                fig = px.imshow(
                    heatmap_data,
                    title=f"Heatmap of {value_col} by {row_cat} and {col_cat}",
                    color_continuous_scale='blues'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Natural language query for visualizations
        st.subheader("Ask for a Specific Visualization")
        nl_query = st.text_input("Describe the visualization you want (e.g., 'Show me billable hours by attorney over time')")
        
        if nl_query and st.button("Generate"):
            st.info("This feature would use natural language processing to interpret your request and generate the appropriate visualization.")
            # Placeholder for NLP-based visualization generation
            # In a full implementation, this would parse the query and create the visualization
    
    with tab3:
        st.header("Attorney Performance Dashboard")
        
        # Check if required columns exist
        if 'User full name (first, last)' in df.columns:
            # Filter for specific attorney
            attorneys = sorted(df['User full name (first, last)'].unique())
            selected_attorney = st.selectbox("Select Attorney", attorneys)
            
            # Filter data for selected attorney
            atty_data = df[df['User full name (first, last)'] == selected_attorney]
            
            # Top metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                if 'Billed hours' in df.columns:
                    total_billed = atty_data['Billed hours'].sum()
                    st.metric("Total Billed Hours", f"{total_billed:.1f}")
            
            with metric_cols[1]:
                if 'Utilization rate' in df.columns:
                    avg_util = atty_data['Utilization rate'].mean()
                    st.metric("Avg Utilization Rate", f"{avg_util:.1f}%")
            
            with metric_cols[2]:
                if 'Billed hours value' in df.columns:
                    total_value = atty_data['Billed hours value'].sum()
                    st.metric("Total Billed Value", f"${total_value:,.2f}")
            
            with metric_cols[3]:
                if 'Matter description' in df.columns:
                    matter_count = atty_data['Matter description'].nunique()
                    st.metric("Total Matters", matter_count)
            
            # Performance charts
            perf_cols = st.columns(2)
            
            with perf_cols[0]:
                if 'Billed hours' in df.columns and 'Activity month' in df.columns:
                    monthly_billed = atty_data.groupby('Activity month')['Billed hours'].sum().reset_index()
                    
                    fig = px.line(
                        monthly_billed,
                        x='Activity month',
                        y='Billed hours',
                        title=f'Monthly Billed Hours - {selected_attorney}',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with perf_cols[1]:
                if 'Practice area' in df.columns and 'Billed hours' in df.columns:
                    practice_hours = atty_data.groupby('Practice area')['Billed hours'].sum().reset_index()
                    practice_hours = practice_hours.sort_values('Billed hours', ascending=False)
                    
                    fig = px.pie(
                        practice_hours,
                        values='Billed hours',
                        names='Practice area',
                        title=f'Hours by Practice Area - {selected_attorney}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Matter table
            if 'Matter description' in df.columns:
                st.subheader(f"Matters Handled by {selected_attorney}")
                
                matter_summary = atty_data.groupby('Matter description').agg({
                    'Billed hours': 'sum',
                    'Billed hours value': 'sum' if 'Billed hours value' in df.columns else None,
                    'Practice area': 'first'
                }).reset_index()
                
                matter_summary = matter_summary.sort_values('Billed hours', ascending=False)
                
                st.dataframe(matter_summary)
        else:
            st.warning("Attorney data is missing from the uploaded file.")
    
    with tab4:
        st.header("Matter Insights")
        
        # Check if required columns exist
        if 'Matter description' in df.columns:
            # Filter for specific matter
            matters = sorted(df['Matter description'].unique())
            selected_matter = st.selectbox("Select Matter", matters)
            
            # Filter data for selected matter
            matter_data = df[df['Matter description'] == selected_matter]
            
            # Get matter metadata
            matter_info_cols = st.columns(4)
            
            with matter_info_cols[0]:
                if 'Practice area' in df.columns:
                    practice = matter_data['Practice area'].iloc[0] if not matter_data['Practice area'].empty else "Unknown"
                    st.metric("Practice Area", practice)
            
            with matter_info_cols[1]:
                if 'Matter status' in df.columns:
                    status = matter_data['Matter status'].iloc[0] if not matter_data['Matter status'].empty else "Unknown"
                    st.metric("Status", status)
            
            with matter_info_cols[2]:
                if 'Responsible attorney' in df.columns:
                    resp_atty = matter_data['Responsible attorney'].iloc[0] if not matter_data['Responsible attorney'].empty else "Unknown"
                    st.metric("Responsible Attorney", resp_atty)
            
            with matter_info_cols[3]:
                if 'Billed hours' in df.columns:
                    total_hours = matter_data['Billed hours'].sum()
                    st.metric("Total Hours", f"{total_hours:.1f}")
            
            # Matter analytics
            matter_cols = st.columns(2)
            
            with matter_cols[0]:
                if 'Billed hours' in df.columns and 'Activity month' in df.columns:
                    monthly_matter = matter_data.groupby('Activity month')['Billed hours'].sum().reset_index()
                    
                    fig = px.bar(
                        monthly_matter,
                        x='Activity month',
                        y='Billed hours',
                        title=f'Monthly Hours - {selected_matter}',
                        color='Billed hours',
                        color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with matter_cols[1]:
                if 'User full name (first, last)' in df.columns and 'Billed hours' in df.columns:
                    atty_contrib = matter_data.groupby('User full name (first, last)')['Billed hours'].sum().reset_index()
                    atty_contrib = atty_contrib.sort_values('Billed hours', ascending=False)
                    
                    fig = px.pie(
                        atty_contrib,
                        values='Billed hours',
                        names='User full name (first, last)',
                        title=f'Attorney Contribution - {selected_matter}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Billable vs non-billable breakdown
            if 'Billed hours' in df.columns and 'Non-billable hours' in df.columns:
                bill_cols = st.columns(2)
                
                with bill_cols[0]:
                    bill_breakdown = pd.DataFrame({
                        'Category': ['Billable', 'Non-Billable'],
                        'Hours': [
                            matter_data['Billed hours'].sum(),
                            matter_data['Non-billable hours'].sum()
                        ]
                    })
                    
                    fig = px.pie(
                        bill_breakdown,
                        values='Hours',
                        names='Category',
                        title='Billable vs Non-Billable Hours',
                        color_discrete_sequence=['#4e8df5', '#a3c4f3']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with bill_cols[1]:
                    if 'Billed hours value' in df.columns and 'Non-billable hours value' in df.columns:
                        value_breakdown = pd.DataFrame({
                            'Category': ['Billable', 'Non-Billable'],
                            'Value': [
                                matter_data['Billed hours value'].sum(),
                                matter_data['Non-billable hours value'].sum()
                            ]
                        })
                        
                        fig = px.pie(
                            value_breakdown,
                            values='Value',
                            names='Category',
                            title='Billable vs Non-Billable Value',
                            color_discrete_sequence=['#4e8df5', '#a3c4f3']
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Matter data is missing from the uploaded file.")

    # Add a feedback section
    st.markdown("---")
    st.subheader("Dashboard Feedback")
    
    feedback = st.text_area("Please provide any feedback on this dashboard prototype:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! In a production version, this would be saved.")

else:
    # Instructions and sample visualization when no file is uploaded
    st.info("Please upload a CSV file to generate the analytics dashboard.")
    
    # Sample visualization as a preview
    st.subheader("Sample Dashboard Preview")
    
    sample_data = {
        'Attorneys': ['Smith, John', 'Jones, Sarah', 'Williams, David', 'Brown, Emily', 'Miller, Michael'],
        'Billed Hours': [120, 145, 95, 110, 130],
        'Utilization': [75, 82, 65, 72, 78]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            sample_df,
            x='Attorneys',
            y='Billed Hours',
            title='Sample: Attorney Billed Hours',
            color='Billed Hours',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            sample_df,
            x='Billed Hours',
            y='Utilization',
            title='Sample: Hours vs Utilization',
            color='Utilization',
            text='Attorneys'
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### What this dashboard will provide:
    - Quick insights into attorney performance
    - Matter profitability analysis
    - Practice area distribution
    - Billable vs. non-billable time tracking
    - Custom visualizations based on your needs
    - And much more!
    
    Upload your Clio data export to get started.
    """)

# Add the Real-Time Connection tab
    with tab5:
        st.header("Real-Time Clio API Connection")
        
        st.markdown("""
        ### Connect to Clio API for Real-Time Data
        
        This section allows you to establish a direct connection to your Clio account for real-time data analysis.
        """)
        
        # Connection configuration
        api_col1, api_col2 = st.columns(2)
        
        with api_col1:
            st.subheader("API Connection")
            st.text_input("Clio API Key", type="password", placeholder="Enter your API key")
            st.text_input("Clio Client ID", placeholder="Enter your client ID")
            st.text_input("Clio Client Secret", type="password", placeholder="Enter your client secret")
            
            if st.button("Test Connection"):
                st.success("Connection simulated successfully! (Demo mode)")
        
        with api_col2:
            st.subheader("Data Refresh Settings")
            refresh_interval = st.selectbox(
                "Auto-refresh interval",
                ["No auto-refresh", "5 minutes", "15 minutes", "30 minutes", "1 hour", "Daily"]
            )
            
            data_to_fetch = st.multiselect(
                "Data to include",
                ["Matters", "Time entries", "Billing", "Contacts", "Calendar events", "Tasks", "Notes", "Documents"],
                default=["Matters", "Time entries", "Billing"]
            )
            
            historical_data = st.slider("Historical data to fetch (months)", 1, 36, 12)
            
            if st.button("Connect & Fetch Data"):
                with st.spinner("Simulating API connection and data fetch..."):
                    # Simulate a delay
                    import time
                    time.sleep(2)
                    st.success("Connection simulated successfully! (Demo mode)")
        
        st.subheader("Real-Time Data Preview")
        st.info("In the fully implemented version, this section would display live data from Clio.")
        
        # Sample real-time preview
        st.markdown("#### Recent Activity (Simulated)")
        
        # Create some sample real-time data
        import datetime
        
        now = datetime.datetime.now()
        
        recent_data = pd.DataFrame({
            'Timestamp': [
                now - datetime.timedelta(minutes=5),
                now - datetime.timedelta(minutes=15),
                now - datetime.timedelta(minutes=42),
                now - datetime.timedelta(hours=1, minutes=12),
                now - datetime.timedelta(hours=2, minutes=34)
            ],
            'Attorney': ['John Smith', 'Sarah Jones', 'John Smith', 'Michael Brown', 'Sarah Jones'],
            'Activity': [
                'Time entry added',
                'Matter updated',
                'Document uploaded',
                'Invoice generated',
                'New matter created'
            ],
            'Matter': [
                'Johnson Litigation',
                'ABC Corp Acquisition',
                'Johnson Litigation',
                'XYZ Estate Planning',
                'Smith Contract Review'
            ]
        })
        
        recent_data['Timestamp'] = recent_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(recent_data, use_container_width=True)
        
        # Real-time metrics
        rt_cols = st.columns(4)
        
        with rt_cols[0]:
            st.metric("Active Matters", "47", "+3")
        
        with rt_cols[1]:
            st.metric("Unbilled Time", "$12,450", "+$1,250")
        
        with rt_cols[2]:
            st.metric("Today's Hours", "24.5", "+2.5")
        
        with rt_cols[3]:
            st.metric("Pending Tasks", "18", "-5")
        
        # Real-time settings
        st.subheader("Real-Time Integration Settings")
        
        rt_settings_col1, rt_settings_col2 = st.columns(2)
        
        with rt_settings_col1:
            st.checkbox("Enable real-time notifications", value=True)
            st.checkbox("Sync calendar events", value=True)
            st.checkbox("Track time entry changes", value=True)
        
        with rt_settings_col2:
            st.checkbox("Monitor billing activities", value=True)
            st.checkbox("Track document uploads", value=False)
            st.checkbox("Alert on deadline changes", value=True)
        
        st.markdown("""
        ### Coming Soon
        - Direct time entry from dashboard
        - Real-time collaboration indicators
        - Mobile notifications and alerts
        - Custom report scheduling
        """)

# Footer
st.markdown("---")
st.markdown("© 2025 Law Firm Analytics Dashboard | Prototype Version")
