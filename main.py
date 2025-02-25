import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import anthropic
import os
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="Law Firm Analytics Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'chart_history' not in st.session_state:
    st.session_state.chart_history = []

# Function to detect column types
def detect_column_types(df):
    date_cols = []
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        # Check if column might be a date
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)
        # Check if column is numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return date_cols, numeric_cols, categorical_cols

# Function to generate LLM visualization recommendation
def get_visualization_recommendation(df, question):
    # Use the API key that's already in session state
    api_key = st.session_state.get('api_key')
    
    if not api_key or api_key == "your-default-api-key-here":
        st.error("No valid API key configured. Please update the application with a valid Anthropic API key.")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)

    # Get column information and first few rows
    column_info = {
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "sample_rows": df.head(3).to_dict(orient='records')
    }
    
    # Create the prompt
    prompt = f"""
    You are an expert data visualization advisor for a law firm analytics dashboard.
    
    Here's information about the dataset:
    {json.dumps(column_info, indent=2)}
    
    The user wants to: {question}
    
    Based on this, please recommend a visualization approach that would best answer their question.
    Provide your response in the following JSON format:
    
    {{
        "chart_type": "One of: bar, line, scatter, pie, histogram, heatmap, box, or area",
        "title": "Suggested title for the chart",
        "x_axis": "Column name for x-axis",
        "y_axis": "Column name for y-axis (or list of columns for multi-series)",
        "color_by": "Optional column to use for color differentiation",
        "facet_by": "Optional column to use for creating small multiples",
        "filters": "Optional filters to apply before visualization",
        "rationale": "Brief explanation of why this visualization works best"
    }}
    
    Only include the JSON in your response, no additional text.
    """
    
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0,
            system="You are an expert data visualization advisor. Provide recommendations in JSON format only.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the JSON response
        try:
            recommendation = json.loads(message.content[0].text)
            return recommendation
        except json.JSONDecodeError:
            st.error("Failed to parse the recommendation. Please try a different question.")
            return None
            
    except Exception as e:
        st.error(f"Error querying Claude API: {str(e)}")
        return None

# Function to create visualization based on recommendation
def create_visualization(df, recommendation):
    chart_type = recommendation.get('chart_type', '').lower()
    x_axis = recommendation.get('x_axis')
    y_axis = recommendation.get('y_axis')
    color_by = recommendation.get('color_by')
    title = recommendation.get('title')
    
    if not x_axis or not y_axis:
        st.warning("Insufficient information to create visualization.")
        return None
    
    try:
        if chart_type == 'bar':
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_by, title=title)
        elif chart_type == 'line':
            fig = px.line(df, x=x_axis, y=y_axis, color=color_by, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=title)
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_axis, values=y_axis, title=title)
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=x_axis, color=color_by, title=title)
        elif chart_type == 'heatmap':
            # For heatmap, we pivot the data
            pivot_data = pd.pivot_table(df, values=y_axis, index=x_axis, columns=color_by)
            fig = px.imshow(pivot_data, title=title)
        elif chart_type == 'box':
            fig = px.box(df, x=x_axis, y=y_axis, color=color_by, title=title)
        elif chart_type == 'area':
            fig = px.area(df, x=x_axis, y=y_axis, color=color_by, title=title)
        else:
            st.warning(f"Chart type '{chart_type}' not supported.")
            return None
            
        fig.update_layout(
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Sidebar for file upload
with st.sidebar:
    st.image("https://www.svgrepo.com/show/491978/scales.svg", width=100)
    st.title("Law Firm Analytics")
    
    # Use environment variable for API key - no user input needed
    # Set a default API key for the prototype
    if 'api_key' not in st.session_state:
        # In a real deployment, you would set this from environment variable
        # st.session_state.api_key = os.environ.get("ANTHROPIC_API_KEY", "your-default-api-key-here")
        st.session_state.api_key = "your-default-api-key-here"  # Replace with actual key in production
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.filtered_data = data
            st.session_state.columns = list(data.columns)
            
            # Detect column types
            date_cols, numeric_cols, categorical_cols = detect_column_types(data)
            st.session_state.date_columns = date_cols
            st.session_state.numeric_columns = numeric_cols
            st.session_state.categorical_columns = categorical_cols
            
            st.success(f"File uploaded successfully with {len(data)} rows and {len(data.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Clio API connection placeholder
    st.markdown("---")
    st.subheader("Clio API Integration")
    st.info("Clio API integration will be available in the full version.")
    
    # Filter settings
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("Filters")
        
        # Date range filter if date columns are available
        if st.session_state.date_columns:
            selected_date_col = st.selectbox("Date Column", st.session_state.date_columns)
            
            if selected_date_col:
                try:
                    df_dates = pd.to_datetime(st.session_state.data[selected_date_col])
                    min_date = df_dates.min().date()
                    max_date = df_dates.max().date()
                    
                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        mask = (df_dates.dt.date >= start_date) & (df_dates.dt.date <= end_date)
                        st.session_state.filtered_data = st.session_state.data[mask]
                except Exception as e:
                    st.warning(f"Could not apply date filter: {str(e)}")
        
        # Category filters
        if st.session_state.categorical_columns:
            selected_cat_col = st.selectbox("Category Filter", st.session_state.categorical_columns)
            
            if selected_cat_col:
                unique_values = st.session_state.filtered_data[selected_cat_col].dropna().unique()
                selected_values = st.multiselect(
                    f"Select {selected_cat_col}",
                    options=unique_values,
                    default=list(unique_values)
                )
                
                if selected_values:
                    st.session_state.filtered_data = st.session_state.filtered_data[
                        st.session_state.filtered_data[selected_cat_col].isin(selected_values)
                    ]
        
        # Numeric range filter
        if st.session_state.numeric_columns:
            selected_num_col = st.selectbox("Numeric Filter", st.session_state.numeric_columns)
            
            if selected_num_col:
                min_val = float(st.session_state.filtered_data[selected_num_col].min())
                max_val = float(st.session_state.filtered_data[selected_num_col].max())
                
                value_range = st.slider(
                    f"{selected_num_col} Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                
                st.session_state.filtered_data = st.session_state.filtered_data[
                    (st.session_state.filtered_data[selected_num_col] >= value_range[0]) & 
                    (st.session_state.filtered_data[selected_num_col] <= value_range[1])
                ]

# Main area
st.title("Law Firm Analytics Dashboard")

# Display data overview
if st.session_state.data is not None:
    # Display tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Attorney Performance", "Practice Area Analysis", "Matter Insights", "Custom Visualizations"])
    
    with tab1:
        st.header("Attorney Performance Dashboard")
        
        # Attorney-specific KPIs
        # Calculate key metrics if columns exist
        df = st.session_state.filtered_data
        
        # Attorney selection
        if 'User full name (first, last)' in df.columns:
            attorneys = sorted(df['User full name (first, last)'].unique())
            selected_attorney = st.selectbox("Select Attorney", ["All Attorneys"] + list(attorneys))
            
            if selected_attorney != "All Attorneys":
                df = df[df['User full name (first, last)'] == selected_attorney]
        
        # KPI metrics section
        st.subheader("Key Performance Indicators")
        
        # First row of KPIs
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            if 'Billed hours' in df.columns:
                total_billed = df['Billed hours'].sum()
                st.metric("Total Billed Hours", f"{total_billed:.1f}")
            else:
                st.metric("Total Billed Hours", "N/A")
                
        with kpi_col2:
            if 'Non-billable hours' in df.columns:
                total_non_billable = df['Non-billable hours'].sum()
                st.metric("Non-Billable Hours", f"{total_non_billable:.1f}")
            else:
                st.metric("Non-Billable Hours", "N/A")
        
        with kpi_col3:
            if 'Utilization rate' in df.columns:
                avg_utilization = df['Utilization rate'].mean()
                st.metric("Avg Utilization Rate", f"{avg_utilization:.1f}%")
            else:
                st.metric("Avg Utilization Rate", "N/A")
        
        with kpi_col4:
            if 'Billed hours value' in df.columns:
                total_billed_value = df['Billed hours value'].sum()
                st.metric("Total Billed Value", f"${total_billed_value:,.2f}")
            else:
                st.metric("Total Billed Value", "N/A")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(st.session_state.filtered_data.head(10), use_container_width=True)
        
        # Column summary
        st.subheader("Column Summary")
        
        col_stats = []
        for col in st.session_state.columns:
            col_type = "Date" if col in st.session_state.date_columns else \
                      "Numeric" if col in st.session_state.numeric_columns else "Categorical"
                      
            # Basic stats
            missing = st.session_state.data[col].isna().sum()
            missing_pct = (missing / len(st.session_state.data)) * 100
            
            # Additional stats based on column type
            if col_type == "Numeric":
                min_val = st.session_state.data[col].min()
                max_val = st.session_state.data[col].max()
                mean_val = st.session_state.data[col].mean()
                additional_info = f"Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}"
            elif col_type == "Categorical":
                unique_vals = st.session_state.data[col].nunique()
                additional_info = f"Unique values: {unique_vals}"
            else:  # Date
                try:
                    date_series = pd.to_datetime(st.session_state.data[col])
                    min_date = date_series.min()
                    max_date = date_series.max()
                    additional_info = f"Range: {min_date.date()} to {max_date.date()}"
                except:
                    additional_info = "Date conversion error"
            
            col_stats.append({
                "Column": col,
                "Type": col_type,
                "Missing": f"{missing} ({missing_pct:.1f}%)",
                "Details": additional_info
            })
        
        st.dataframe(pd.DataFrame(col_stats), use_container_width=True)
        
        # Attorney performance charts
        st.subheader("Performance Analysis")
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Utilization Rate Over Time
        with chart_col1:
            if all(col in df.columns for col in ['Activity date', 'Utilization rate']):
                try:
                    # Convert to datetime if it's not already
                    df['Activity date'] = pd.to_datetime(df['Activity date'])
                    
                    # Group by month and calculate average utilization rate
                    monthly_util = df.groupby(df['Activity date'].dt.strftime('%Y-%m'))[['Utilization rate']].mean().reset_index()
                    monthly_util = monthly_util.sort_values('Activity date')
                    
                    fig = px.line(
                        monthly_util, 
                        x="Activity date", 
                        y="Utilization rate",
                        title="Monthly Utilization Rate",
                        labels={"Utilization rate": "Utilization %", "Activity date": "Month"},
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating utilization chart: {str(e)}")
            else:
                st.info("Utilization rate data not available.")
        
        # Billable vs Non-billable Hours
        with chart_col2:
            if all(col in df.columns for col in ['Billed hours', 'Non-billable hours']):
                try:
                    # Create a summary dataframe
                    hours_summary = pd.DataFrame({
                        'Category': ['Billable Hours', 'Non-billable Hours'],
                        'Hours': [df['Billed hours'].sum(), df['Non-billable hours'].sum()]
                    })
                    
                    fig = px.pie(
                        hours_summary,
                        values='Hours',
                        names='Category',
                        title="Billable vs Non-billable Hours Distribution",
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating hours distribution chart: {str(e)}")
            else:
                st.info("Billable and non-billable hours data not available.")
        
        # Attorney Time Tracking Analysis
        st.subheader("Time Tracking Analysis")
        
        time_col1, time_col2 = st.columns(2)
        
        # Hours by Practice Area
        with time_col1:
            if all(col in df.columns for col in ['Practice area', 'Billed hours']):
                try:
                    # Group by practice area
                    practice_hours = df.groupby('Practice area')[['Billed hours']].sum().reset_index()
                    practice_hours = practice_hours.sort_values('Billed hours', ascending=False)
                    
                    fig = px.bar(
                        practice_hours,
                        x='Practice area',
                        y='Billed hours',
                        title="Billable Hours by Practice Area",
                        color='Billed hours',
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating practice area chart: {str(e)}")
            else:
                st.info("Practice area and billable hours data not available.")
        
        # Monthly Hours Trend
        with time_col2:
            if all(col in df.columns for col in ['Activity date', 'Billed hours', 'Non-billable hours']):
                try:
                    # Convert to datetime if it's not already
                    df['Activity date'] = pd.to_datetime(df['Activity date'])
                    
                    # Group by month and calculate hours
                    monthly_hours = df.groupby(df['Activity date'].dt.strftime('%Y-%m'))[['Billed hours', 'Non-billable hours']].sum().reset_index()
                    monthly_hours = monthly_hours.sort_values('Activity date')
                    
                    fig = px.bar(
                        monthly_hours,
                        x='Activity date',
                        y=['Billed hours', 'Non-billable hours'],
                        title="Monthly Hours Breakdown",
                        labels={"value": "Hours", "Activity date": "Month", "variable": "Category"},
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating monthly hours chart: {str(e)}")
            else:
                st.info("Monthly hours data not available.")
    
    with tab2:
        st.header("Practice Area Analysis")
        
        df = st.session_state.filtered_data
        
        # Practice area selection
        if 'Practice area' in df.columns:
            practice_areas = sorted(df['Practice area'].unique())
            selected_practice = st.selectbox("Select Practice Area", ["All Practice Areas"] + list(practice_areas), key="practice_select")
            
            if selected_practice != "All Practice Areas":
                df = df[df['Practice area'] == selected_practice]
        
        # Practice area KPIs
        st.subheader("Practice Area Metrics")
        
        pa_kpi_col1, pa_kpi_col2, pa_kpi_col3, pa_kpi_col4 = st.columns(4)
        
        with pa_kpi_col1:
            if 'Billed hours' in df.columns:
                total_pa_billed = df['Billed hours'].sum()
                st.metric("Total Billed Hours", f"{total_pa_billed:.1f}")
            else:
                st.metric("Total Billed Hours", "N/A")
                
        with pa_kpi_col2:
            if 'Billed hours value' in df.columns:
                avg_hourly_rate = df['Billed hours value'].sum() / max(df['Billed hours'].sum(), 1)
                st.metric("Avg. Hourly Rate", f"${avg_hourly_rate:.2f}")
            else:
                st.metric("Avg. Hourly Rate", "N/A")
        
        with pa_kpi_col3:
            if 'Matter description' in df.columns:
                matter_count = df['Matter description'].nunique()
                st.metric("Active Matters", f"{matter_count}")
            else:
                st.metric("Active Matters", "N/A")
        
        with pa_kpi_col4:
            if 'User full name (first, last)' in df.columns:
                attorney_count = df['User full name (first, last)'].nunique()
                st.metric("Attorneys Involved", f"{attorney_count}")
            else:
                st.metric("Attorneys Involved", "N/A")
                
        # Practice area charts
        st.subheader("Practice Area Performance")
        
        pa_chart_col1, pa_chart_col2 = st.columns(2)
        
        # Top Matters by Hours
        with pa_chart_col1:
            if all(col in df.columns for col in ['Matter description', 'Billed hours']):
                try:
                    # Group by matter
                    matter_hours = df.groupby('Matter description')[['Billed hours']].sum().reset_index()
                    matter_hours = matter_hours.sort_values('Billed hours', ascending=False).head(10)
                    
                    fig = px.bar(
                        matter_hours,
                        x='Billed hours',
                        y='Matter description',
                        title="Top 10 Matters by Billable Hours",
                        orientation='h',
                        color='Billed hours',
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating top matters chart: {str(e)}")
            else:
                st.info("Matter and billable hours data not available.")
        
        # Top Attorneys in Practice Area
        with pa_chart_col2:
            if all(col in df.columns for col in ['User full name (first, last)', 'Billed hours']):
                try:
                    # Group by attorney
                    attorney_hours = df.groupby('User full name (first, last)')[['Billed hours']].sum().reset_index()
                    attorney_hours = attorney_hours.sort_values('Billed hours', ascending=False).head(10)
                    
                    fig = px.bar(
                        attorney_hours,
                        x='Billed hours',
                        y='User full name (first, last)',
                        title="Top 10 Attorneys by Billable Hours",
                        orientation='h',
                        color='Billed hours',
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating top attorneys chart: {str(e)}")
            else:
                st.info("Attorney and billable hours data not available.")
        
        # Monthly Trend for Practice Area
        st.subheader("Monthly Trends")
        
        if all(col in df.columns for col in ['Activity date', 'Billed hours', 'Utilization rate']):
            try:
                # Convert to datetime if it's not already
                df['Activity date'] = pd.to_datetime(df['Activity date'])
                
                # Group by month
                monthly_data = df.groupby(df['Activity date'].dt.strftime('%Y-%m')).agg({
                    'Billed hours': 'sum',
                    'Utilization rate': 'mean',
                    'Billed hours value': 'sum'
                }).reset_index()
                monthly_data = monthly_data.sort_values('Activity date')
                
                # Create figure with secondary y-axis
                fig = go.Figure()
                
                # Add traces
                fig.add_trace(
                    go.Bar(
                        x=monthly_data['Activity date'],
                        y=monthly_data['Billed hours'],
                        name="Billed Hours",
                        marker_color='rgb(55, 83, 109)'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=monthly_data['Activity date'],
                        y=monthly_data['Utilization rate'],
                        name="Utilization Rate",
                        marker_color='rgb(26, 118, 255)',
                        mode='lines+markers',
                        yaxis="y2"
                    )
                )
                
                # Create axis objects
                fig.update_layout(
                    title="Monthly Performance Trends",
                    xaxis=dict(title="Month"),
                    yaxis=dict(
                        title="Billed Hours",
                        titlefont=dict(color="rgb(55, 83, 109)"),
                        tickfont=dict(color="rgb(55, 83, 109)")
                    ),
                    yaxis2=dict(
                        title="Utilization Rate (%)",
                        titlefont=dict(color="rgb(26, 118, 255)"),
                        tickfont=dict(color="rgb(26, 118, 255)"),
                        anchor="x",
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating monthly trends chart: {str(e)}")
        else:
            st.info("Monthly trend data not available.")
            
    with tab3:
        st.header("Matter Insights")
        
        df = st.session_state.filtered_data
        
        # Matter selection
        if 'Matter description' in df.columns:
            matters = sorted(df['Matter description'].unique())
            selected_matter = st.selectbox("Select Matter", ["All Matters"] + list(matters), key="matter_select")
            
            if selected_matter != "All Matters":
                df = df[df['Matter description'] == selected_matter]
        
        # Matter KPIs
        st.subheader("Matter Metrics")
        
        matter_kpi_col1, matter_kpi_col2, matter_kpi_col3, matter_kpi_col4 = st.columns(4)
        
        with matter_kpi_col1:
            if 'Billed hours' in df.columns:
                total_matter_billed = df['Billed hours'].sum()
                st.metric("Total Billed Hours", f"{total_matter_billed:.1f}")
            else:
                st.metric("Total Billed Hours", "N/A")
                
        with matter_kpi_col2:
            if 'Billed hours value' in df.columns:
                total_billings = df['Billed hours value'].sum()
                st.metric("Total Billings", f"${total_billings:,.2f}")
            else:
                st.metric("Total Billings", "N/A")
        
        with matter_kpi_col3:
            if 'User full name (first, last)' in df.columns:
                attorney_count = df['User full name (first, last)'].nunique()
                st.metric("Attorneys on Matter", f"{attorney_count}")
            else:
                st.metric("Attorneys on Matter", "N/A")
        
        with matter_kpi_col4:
            if 'Matter status' in df.columns and selected_matter != "All Matters":
                matter_status = df['Matter status'].iloc[0] if not df.empty else "Unknown"
                st.metric("Matter Status", matter_status)
            else:
                st.metric("Matter Status", "Various" if selected_matter == "All Matters" else "N/A")
                
        # Matter charts
        st.subheader("Matter Analysis")
        
        matter_chart_col1, matter_chart_col2 = st.columns(2)
        
        # Hours by Attorney
        with matter_chart_col1:
            if all(col in df.columns for col in ['User full name (first, last)', 'Billed hours']):
                try:
                    # Group by attorney
                    attorney_matter_hours = df.groupby('User full name (first, last)')[['Billed hours']].sum().reset_index()
                    attorney_matter_hours = attorney_matter_hours.sort_values('Billed hours', ascending=False)
                    
                    fig = px.pie(
                        attorney_matter_hours,
                        values='Billed hours',
                        names='User full name (first, last)',
                        title="Billable Hours Distribution by Attorney",
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating attorney distribution chart: {str(e)}")
            else:
                st.info("Attorney and billable hours data not available.")
        
        # Billable vs Non-billable for Matter
        with matter_chart_col2:
            if all(col in df.columns for col in ['Billed hours', 'Non-billable hours']):
                try:
                    # Create a summary dataframe
                    matter_hours_summary = pd.DataFrame({
                        'Category': ['Billable Hours', 'Non-billable Hours'],
                        'Hours': [df['Billed hours'].sum(), df['Non-billable hours'].sum()]
                    })
                    
                    fig = px.bar(
                        matter_hours_summary,
                        x='Category',
                        y='Hours',
                        title="Billable vs Non-billable Hours",
                        color='Category',
                        color_discrete_map={
                            'Billable Hours': 'rgb(26, 118, 255)',
                            'Non-billable Hours': 'rgb(166, 189, 219)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating hours distribution chart: {str(e)}")
            else:
                st.info("Billable and non-billable hours data not available.")
        
        # Matter Timeline
        st.subheader("Matter Timeline")
        
        if all(col in df.columns for col in ['Activity date', 'Billed hours']):
            try:
                # Convert to datetime if it's not already
                df['Activity date'] = pd.to_datetime(df['Activity date'])
                
                # Group by date
                daily_hours = df.groupby(df['Activity date'].dt.strftime('%Y-%m-%d'))[['Billed hours', 'Non-billable hours']].sum().reset_index()
                daily_hours = daily_hours.sort_values('Activity date')
                
                fig = px.line(
                    daily_hours,
                    x='Activity date',
                    y=['Billed hours', 'Non-billable hours'],
                    title="Daily Hours Timeline",
                    labels={"value": "Hours", "Activity date": "Date", "variable": "Category"},
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating timeline chart: {str(e)}")
        else:
            st.info("Timeline data not available.")
        
        # Matter additional information
        if selected_matter != "All Matters":
            st.subheader("Matter Details")
            
            matter_info_col1, matter_info_col2 = st.columns(2)
            
            with matter_info_col1:
                if 'Matter status' in df.columns:
                    st.markdown(f"**Status:** {df['Matter status'].iloc[0] if not df.empty else 'Unknown'}")
                
                if 'Practice area' in df.columns:
                    st.markdown(f"**Practice Area:** {df['Practice area'].iloc[0] if not df.empty else 'Unknown'}")
                
                if 'Responsible attorney' in df.columns:
                    st.markdown(f"**Responsible Attorney:** {df['Responsible attorney'].iloc[0] if not df.empty else 'Unknown'}")
            
            with matter_info_col2:
                if 'Originating attorney' in df.columns:
                    st.markdown(f"**Originating Attorney:** {df['Originating attorney'].iloc[0] if not df.empty else 'Unknown'}")
                
                if 'Matter close date' in df.columns:
                    close_date = df['Matter close date'].iloc[0] if not df.empty else 'Unknown'
                    st.markdown(f"**Close Date:** {close_date}")
                
                if 'Billable matter' in df.columns:
                    billable = "Yes" if df['Billable matter'].iloc[0] == 1 else "No" if not df.empty else "Unknown"
                    st.markdown(f"**Billable Matter:** {billable}")
    
    with tab4:
        st.header("Custom Visualizations")
        
        # LLM-guided visualization
        st.subheader("AI-Guided Visualization")
        
        question = st.text_input(
            "What would you like to visualize? Describe in natural language",
            placeholder="E.g., Show me the trend of billable hours over time by practice area"
        )
        
        if question and st.button("Generate Visualization"):
            with st.spinner("Analyzing your data and generating visualization..."):
                recommendation = get_visualization_recommendation(
                    st.session_state.filtered_data,
                    question
                )
                
                if recommendation:
                    st.success("Visualization recommendation created!")
                    
                    # Display the recommendation
                    with st.expander("View recommendation details"):
                        st.json(recommendation)
                    
                    # Create and display the visualization
                    fig = create_visualization(st.session_state.filtered_data, recommendation)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Option to save chart
                        if st.button("Save to Dashboard"):
                            chart_entry = {
                                "name": recommendation.get("title", "Untitled Chart"),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "recommendation": recommendation,
                                "query": question
                            }
                            st.session_state.chart_history.append(chart_entry)
                            st.success("Chart saved to dashboard!")
        
        # Manual chart builder
        st.subheader("Manual Chart Builder")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"]
            )
            
            x_axis = st.selectbox("X-Axis", st.session_state.columns)
            
            if chart_type != "Pie Chart":
                y_axis = st.selectbox("Y-Axis", st.session_state.numeric_columns if st.session_state.numeric_columns else st.session_state.columns)
            else:
                y_axis = st.selectbox("Values", st.session_state.numeric_columns if st.session_state.numeric_columns else st.session_state.columns)
            
            color_by = st.selectbox("Color By (Optional)", ["None"] + st.session_state.categorical_columns)
            color_by = None if color_by == "None" else color_by
            
            chart_title = st.text_input("Chart Title", f"{chart_type} of {y_axis} by {x_axis}")
        
        with chart_col2:
            try:
                if chart_type == "Bar Chart":
                    fig = px.bar(
                        st.session_state.filtered_data,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=chart_title
                    )
                elif chart_type == "Line Chart":
                    fig = px.line(
                        st.session_state.filtered_data,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=chart_title
                    )
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(
                        st.session_state.filtered_data,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=chart_title
                    )
                elif chart_type == "Pie Chart":
                    fig = px.pie(
                        st.session_state.filtered_data,
                        names=x_axis,
                        values=y_axis,
                        title=chart_title
                    )
                elif chart_type == "Box Plot":
                    fig = px.box(
                        st.session_state.filtered_data,
                        x=x_axis,
                        y=y_axis,
                        color=color_by,
                        title=chart_title
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save option
                if st.button("Save this chart"):
                    chart_entry = {
                        "name": chart_title,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": chart_type,
                        "x_axis": x_axis,
                        "y_axis": y_axis,
                        "color_by": color_by
                    }
                    st.session_state.chart_history.append(chart_entry)
                    st.success("Chart saved to dashboard!")
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
    
    with tab3:
        st.header("Saved Charts")
        
        if not st.session_state.chart_history:
            st.info("No saved charts yet. Create and save charts from the Custom Visualizations tab.")
        else:
            # Display saved charts in a grid
            for i in range(0, len(st.session_state.chart_history), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(st.session_state.chart_history):
                        chart = st.session_state.chart_history[i + j]
                        
                        with cols[j]:
                            st.subheader(chart["name"])
                            st.caption(f"Created: {chart['timestamp']}")
                            
                            try:
                                # Recreate chart based on type
                                if "recommendation" in chart:
                                    # LLM-recommended chart
                                    fig = create_visualization(
                                        st.session_state.filtered_data,
                                        chart["recommendation"]
                                    )
                                else:
                                    # Manually created chart
                                    chart_type = chart["type"]
                                    
                                    if chart_type == "Bar Chart":
                                        fig = px.bar(
                                            st.session_state.filtered_data,
                                            x=chart["x_axis"],
                                            y=chart["y_axis"],
                                            color=chart["color_by"],
                                            title=chart["name"]
                                        )
                                    elif chart_type == "Line Chart":
                                        fig = px.line(
                                            st.session_state.filtered_data,
                                            x=chart["x_axis"],
                                            y=chart["y_axis"],
                                            color=chart["color_by"],
                                            title=chart["name"]
                                        )
                                    elif chart_type == "Scatter Plot":
                                        fig = px.scatter(
                                            st.session_state.filtered_data,
                                            x=chart["x_axis"],
                                            y=chart["y_axis"],
                                            color=chart["color_by"],
                                            title=chart["name"]
                                        )
                                    elif chart_type == "Pie Chart":
                                        fig = px.pie(
                                            st.session_state.filtered_data,
                                            names=chart["x_axis"],
                                            values=chart["y_axis"],
                                            title=chart["name"]
                                        )
                                    elif chart_type == "Box Plot":
                                        fig = px.box(
                                            st.session_state.filtered_data,
                                            x=chart["x_axis"],
                                            y=chart["y_axis"],
                                            color=chart["color_by"],
                                            title=chart["name"]
                                        )
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Option to remove chart
                                    if st.button(f"Remove", key=f"remove_{i}_{j}"):
                                        st.session_state.chart_history.pop(i + j)
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error recreating chart: {str(e)}")
else:
    # No data uploaded yet
    st.info("👈 Please upload a CSV file to get started.")
    
    # Sample data explanation
    st.markdown("""
    ## How to use this dashboard
    
    1. **Upload Data**: Start by uploading a CSV file with your law firm data.
    2. **Explore Data**: View summary statistics and quick visualizations.
    3. **Create Custom Charts**: Use the AI assistant to generate visualizations or build them manually.
    4. **Save to Dashboard**: Save your favorite charts to create a custom dashboard.
    
    ### Example data you can use
    Upload CSV files containing information such as:
    - Client billing records
    - Case outcomes
    - Staff productivity metrics
    - Practice area performance
    
    ### Coming soon
    - Direct Clio API integration
    - Automated insights and recommendations
    - Export and sharing capabilities
    - Scheduled reports
    """)

# Footer
st.markdown("---")
col1, col2 = st.columns([4, 1])
with col1:
    st.caption("Law Firm Analytics Dashboard Prototype | Powered by Streamlit and Claude")
with col2:
    st.caption("v0.1.0")
