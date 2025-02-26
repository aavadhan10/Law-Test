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

# Function to use Claude to analyze the CSV and identify relevant columns
def analyze_csv_with_llm(df):
    api_key = st.session_state.get('api_key')
    
    if not api_key or api_key == "your-default-api-key-here":
        st.error("No valid API key configured. Please update the application with a valid Anthropic API key.")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Create a sample of the data for Claude to analyze
    sample_rows = min(5, len(df))
    data_sample = df.head(sample_rows).to_csv(index=False)
    columns_info = "\n".join([f"- {col} (Type: {df[col].dtype}, Sample: {df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 'NA'})" for col in df.columns])
    
    # Create the prompt
    prompt = f"""
    I need you to analyze this CSV data from a law firm billing or practice management system.
    
    Here are the columns in the CSV:
    {columns_info}
    
    Here's a sample of the data:
    {data_sample}
    
    Please analyze this data and identify which columns correspond to the following important legal billing concepts:
    
    1. Client information (client name, client ID, etc.)
    2. Matter information (matter description, matter ID, matter status)
    3. Timekeeper information (attorney name, timekeeper ID)
    4. Time entries (dates, hours worked, billable vs. non-billable)
    5. Billing information (rates, amounts, values)
    6. Categorization (practice area, case type, etc.)
    
    For each concept, tell me which column or columns in the CSV contain this information.
    
    Also, please suggest 3-5 interesting visualizations that would be valuable for law firm partners based on this specific data.
    
    Finally, identify what type of legal data this appears to be (e.g., time entries, matter summary, client profitability, etc.).
    
    Provide your response in JSON format like this:
    {{
        "data_type": "brief description of what this data represents",
        "column_mapping": {{
            "client": ["column name(s) for client info"],
            "matter": ["column name(s) for matter info"],
            "timekeeper": ["column name(s) for timekeeper info"],
            "time_entries": ["column name(s) for time entries"],
            "billing": ["column name(s) for billing info"],
            "categorization": ["column name(s) for categorization"]
        }},
        "suggested_visualizations": [
            {{
                "title": "Visualization title",
                "description": "Brief description of what this visualization would show"
            }}
        ]
    }}
    
    Only return the JSON, nothing else.
    """
    
    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0,
            system="You are an expert legal data analyst who specializes in analyzing law firm data. Provide responses in JSON format only.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and parse the JSON response
        try:
            analysis = json.loads(message.content[0].text)
            return analysis
        except json.JSONDecodeError:
            st.error("Failed to parse the analysis. Please try again or contact support.")
            return None
            
    except Exception as e:
        st.error(f"Error querying Claude API: {str(e)}")
        return None

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
            
            # Reset analysis state
            st.session_state.analyzed = False
            st.session_state.llm_analysis_complete = False
            st.session_state.dashboard_ready = False
            
            st.success(f"File uploaded successfully with {len(data)} rows and {len(data.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Clio API connection placeholder
    st.markdown("---")
    st.subheader("Clio API Integration")
    st.info("Clio API integration will be available in the full version.")
    
    # Saved Charts Section in sidebar
    if st.session_state.chart_history:
        st.markdown("---")
        st.subheader("Saved Charts")
        for i, chart in enumerate(st.session_state.chart_history):
            st.write(f"{i+1}. {chart.get('name', 'Unnamed Chart')}")
        
        st.info("View your saved charts in the Dashboard tab.")

# Main area
st.title("Law Firm Analytics Dashboard")

# Display data overview
if st.session_state.data is not None:
    # Step 1: Initial data analysis
    if 'analyzed' not in st.session_state or not st.session_state.analyzed:
        st.header("Step 1: CSV Data Analysis")
        
        st.write("Let's understand what's in your data:")
        
        # Show data overview
        st.subheader("Data Sample")
        st.dataframe(st.session_state.data.head(5), use_container_width=True)
        
        # Column information
        st.subheader("Column Information")
        col_info = []
        for col in st.session_state.data.columns:
            col_type = "Date" if col in st.session_state.date_columns else \
                      "Numeric" if col in st.session_state.numeric_columns else "Text"
            
            non_null = st.session_state.data[col].count()
            total_rows = len(st.session_state.data)
            completeness = (non_null / total_rows) * 100
            
            if col_type == "Numeric":
                stats = f"Min: {st.session_state.data[col].min()}, Max: {st.session_state.data[col].max()}, Avg: {st.session_state.data[col].mean():.2f}"
            elif col_type == "Text":
                unique = st.session_state.data[col].nunique()
                stats = f"Unique values: {unique} ({(unique/total_rows*100):.1f}% of total)"
            else:  # Date
                try:
                    date_range = pd.to_datetime(st.session_state.data[col])
                    stats = f"Range: {date_range.min().date()} to {date_range.max().date()}"
                except:
                    stats = "Unable to parse dates"
            
            col_info.append({
                "Column": col,
                "Type": col_type,
                "Completeness": f"{completeness:.1f}%",
                "Statistics": stats
            })
        
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)
        
        # Button to continue to LLM analysis
        analyze_button = st.button("Analyze Data with AI")
        if analyze_button:
            # Set the session state flag and immediately run the analysis
            st.session_state.analyzed = True
            
            # Show a message to the user
            st.success("Starting AI analysis... Please wait.")
            
            # Run the analysis right away instead of waiting for rerun
            with st.spinner("Claude is analyzing your legal data..."):
                llm_analysis = analyze_csv_with_llm(st.session_state.data)
                
                if llm_analysis:
                    st.session_state.llm_analysis = llm_analysis
                    st.session_state.llm_analysis_complete = True
                    
                    # Extract column mapping
                    st.session_state.column_mapping = {}
                    for category, columns in llm_analysis.get('column_mapping', {}).items():
                        for col in columns:
                            if col in st.session_state.data.columns:
                                st.session_state.column_mapping[category + "_" + col] = col
                    
                    # Save suggested visualizations
                    st.session_state.suggested_visualizations = llm_analysis.get('suggested_visualizations', [])
                    
                    # Identify data type
                    st.session_state.data_type = llm_analysis.get('data_type', 'Legal billing data')
                    
                    st.success("AI analysis complete!")
                    
                    # Display the analysis results
                    st.header("AI Insights")
                    
                    # Display data type
                    st.subheader("Data Type Identified")
                    st.info(llm_analysis.get('data_type', 'Legal billing data'))
                    
                    # Display column mapping
                    st.subheader("Column Mapping")
                    
                    col_map = llm_analysis.get('column_mapping', {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Client Information")
                        for col in col_map.get('client', []):
                            st.write(f"- {col}")
                        
                        st.markdown("##### Matter Information")
                        for col in col_map.get('matter', []):
                            st.write(f"- {col}")
                        
                        st.markdown("##### Timekeeper Information")
                        for col in col_map.get('timekeeper', []):
                            st.write(f"- {col}")
                    
                    with col2:
                        st.markdown("##### Time Entries")
                        for col in col_map.get('time_entries', []):
                            st.write(f"- {col}")
                        
                        st.markdown("##### Billing Information")
                        for col in col_map.get('billing', []):
                            st.write(f"- {col}")
                        
                        st.markdown("##### Categorization")
                        for col in col_map.get('categorization', []):
                            st.write(f"- {col}")
                    
                    # Display suggested visualizations
                    st.subheader("Suggested Visualizations")
                    
                    viz_suggestions = llm_analysis.get('suggested_visualizations', [])
                    
                    for i, viz in enumerate(viz_suggestions):
                        with st.expander(f"{i+1}. {viz.get('title', 'Visualization')}"):
                            st.write(viz.get('description', 'No description provided.'))
                    
                    # Button to continue to dashboard
                    if st.button("Continue to Dashboard"):
                        st.session_state.dashboard_ready = True
                        st.experimental_rerun()
                else:
                    st.error("Could not complete AI analysis. Please try again.")
    
    # Step 2: Show the AI analysis results
    elif 'llm_analysis_complete' in st.session_state and st.session_state.llm_analysis_complete and ('dashboard_ready' not in st.session_state or not st.session_state.dashboard_ready):
        st.header("AI Insights")
        
        llm_analysis = st.session_state.llm_analysis
        
        # Display data type
        st.subheader("Data Type Identified")
        st.info(llm_analysis.get('data_type', 'Legal billing data'))
        
        # Display column mapping
        st.subheader("Column Mapping")
        
        col_map = llm_analysis.get('column_mapping', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Client Information")
            for col in col_map.get('client', []):
                st.write(f"- {col}")
            
            st.markdown("##### Matter Information")
            for col in col_map.get('matter', []):
                st.write(f"- {col}")
            
            st.markdown("##### Timekeeper Information")
            for col in col_map.get('timekeeper', []):
                st.write(f"- {col}")
        
        with col2:
            st.markdown("##### Time Entries")
            for col in col_map.get('time_entries', []):
                st.write(f"- {col}")
            
            st.markdown("##### Billing Information")
            for col in col_map.get('billing', []):
                st.write(f"- {col}")
            
            st.markdown("##### Categorization")
            for col in col_map.get('categorization', []):
                st.write(f"- {col}")
        
        # Display suggested visualizations
        st.subheader("Suggested Visualizations")
        
        viz_suggestions = llm_analysis.get('suggested_visualizations', [])
        
        for i, viz in enumerate(viz_suggestions):
            with st.expander(f"{i+1}. {viz.get('title', 'Visualization')}"):
                st.write(viz.get('description', 'No description provided.'))
        
        # Button to continue to dashboard
        if st.button("Continue to Dashboard"):
            st.session_state.dashboard_ready = True
            st.experimental_rerun()
    
    # Step 3: Show the actual dashboard
    else:
        # Display tabs for the dashboard
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Explorer", "Suggested Visualizations", "Custom Visualizations"])
        
        with tab1:
            st.header("Data Overview Dashboard")
            
            # Display data type and summary
            st.subheader("About This Data")
            st.write(st.session_state.data_type)
            
            # Key metrics based on data type
            st.subheader("Key Metrics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            df = st.session_state.filtered_data
            
            # Dynamically determine which metrics to show based on available columns
            col_map = st.session_state.llm_analysis.get('column_mapping', {})
            
            with metrics_col1:
                # Try to show total records
                st.metric("Total Records", f"{len(df):,}")
            
            with metrics_col2:
                # Try to show unique matters or clients
                if col_map.get('matter'):
                    matter_col = col_map.get('matter')[0]
                    if matter_col in df.columns:
                        unique_matters = df[matter_col].nunique()
                        st.metric("Unique Matters", f"{unique_matters:,}")
                elif col_map.get('client'):
                    client_col = col_map.get('client')[0]
                    if client_col in df.columns:
                        unique_clients = df[client_col].nunique()
                        st.metric("Unique Clients", f"{unique_clients:,}")
            
            with metrics_col3:
                # Try to show total hours or amount
                hour_cols = [col for col in df.columns if 'hour' in col.lower()]
                amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'fee', 'bill'])]
                
                if hour_cols:
                    total_hours = df[hour_cols[0]].sum()
                    st.metric("Total Hours", f"{total_hours:,.1f}")
                elif amount_cols:
                    total_amount = df[amount_cols[0]].sum()
                    st.metric("Total Amount", f"${total_amount:,.2f}")
            
            with metrics_col4:
                # Try to show unique timekeepers
                if col_map.get('timekeeper'):
                    timekeeper_col = col_map.get('timekeeper')[0]
                    if timekeeper_col in df.columns:
                        unique_timekeepers = df[timekeeper_col].nunique()
                        st.metric("Timekeepers", f"{unique_timekeepers:,}")
            
            # Quick summary visualizations
            st.subheader("Data Summary")
            
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                # Try to create categorical breakdown
                category_cols = col_map.get('categorization', [])
                if category_cols and category_cols[0] in df.columns:
                    cat_col = category_cols[0]
                    cat_counts = df[cat_col].value_counts().reset_index()
                    cat_counts.columns = [cat_col, 'Count']
                    
                    try:
                        fig = px.pie(
                            cat_counts, 
                            values='Count', 
                            names=cat_col,
                            title=f"Breakdown by {cat_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create chart: {str(e)}")
            
            with summary_col2:
                # Try to create time-based visualization
                date_cols = col_map.get('time_entries', [])
                date_cols = [col for col in date_cols if col in st.session_state.date_columns]
                
                if date_cols and date_cols[0] in df.columns:
                    date_col = date_cols[0]
                    
                    try:
                        df['date_parsed'] = pd.to_datetime(df[date_col])
                        df['month'] = df['date_parsed'].dt.strftime('%Y-%m')
                        
                        # Aggregate by month
                        monthly_data = df.groupby('month').size().reset_index()
                        monthly_data.columns = ['Month', 'Count']
                        
                        fig = px.line(
                            monthly_data,
                            x='Month',
                            y='Count',
                            title="Monthly Activity",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create time chart: {str(e)}")
                        
            # Data filters section
            st.subheader("Data Filters")
            
            filter_container = st.container()
            
            with filter_container:
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                filtered_data = df.copy()
                
                # Try to add date filter
                with filter_col1:
                    if date_cols and date_cols[0] in df.columns:
                        date_col = date_cols[0]
                        st.subheader("Date Filter")
                        
                        try:
                            df['date_parsed'] = pd.to_datetime(df[date_col])
                            min_date = df['date_parsed'].min().date()
                            max_date = df['date_parsed'].max().date()
                            
                            date_range = st.date_input(
                                "Select Date Range",
                                value=(min_date, max_date),
                                min_value=min_date,
                                max_value=max_date
                            )
                            
                            if len(date_range) == 2:
                                start_date, end_date = date_range
                                filtered_data = filtered_data[
                                    (filtered_data['date_parsed'].dt.date >= start_date) & 
                                    (filtered_data['date_parsed'].dt.date <= end_date)
                                ]
                        except Exception as e:
                            st.error(f"Could not apply date filter: {str(e)}")
                
                # Try to add category filter
                with filter_col2:
                    if category_cols and category_cols[0] in df.columns:
                        cat_col = category_cols[0]
                        st.subheader("Category Filter")
                        
                        unique_cats = df[cat_col].dropna().unique()
                        selected_cats = st.multiselect(
                            f"Select {cat_col}",
                            options=sorted(unique_cats),
                            default=list(unique_cats)[:min(5, len(unique_cats))]
                        )
                        
                        if selected_cats:
                            filtered_data = filtered_data[filtered_data[cat_col].isin(selected_cats)]
                
                # Try to add client/matter filter
                with filter_col3:
                    client_cols = col_map.get('client', [])
                    matter_cols = col_map.get('matter', [])
                    
                    if client_cols and client_cols[0] in df.columns:
                        client_col = client_cols[0]
                        st.subheader("Client Filter")
                        
                        unique_clients = df[client_col].dropna().unique()
                        selected_client = st.selectbox(
                            f"Select {client_col}",
                            options=["All Clients"] + sorted(unique_clients)
                        )
                        
                        if selected_client != "All Clients":
                            filtered_data = filtered_data[filtered_data[client_col] == selected_client]
                    elif matter_cols and matter_cols[0] in df.columns:
                        matter_col = matter_cols[0]
                        st.subheader("Matter Filter")
                        
                        unique_matters = df[matter_col].dropna().unique()
                        selected_matter = st.selectbox(
                            f"Select {matter_col}",
                            options=["All Matters"] + sorted(unique_matters)[:100]  # Limit to 100 for performance
                        )
                        
                        if selected_matter != "All Matters":
                            filtered_data = filtered_data[filtered_data[matter_col] == selected_matter]
                
                # Update the filtered data
                st.session_state.filtered_data = filtered_data
                
                # Show filtered records count
                st.metric("Filtered Records", f"{len(filtered_data):,} ({len(filtered_data)/len(df)*100:.1f}% of total)")
        
        with tab2:
            st.header("Data Explorer")
            
            df = st.session_state.filtered_data
            
            # Column selector
            st.subheader("Column Explorer")
            
            explorer_col1, explorer_col2 = st.columns([1, 3])
            
            with explorer_col1:
                selected_column = st.selectbox("Select Column to Explore", st.session_state.columns)
                
                if selected_column:
                    col_type = "Date" if selected_column in st.session_state.date_columns else \
                            "Numeric" if selected_column in st.session_state.numeric_columns else "Text"
                    
                    st.write(f"**Type:** {col_type}")
                    
                    non_null = df[selected_column].count()
                    total_rows = len(df)
                    completeness = (non_null / total_rows) * 100
                    
                    st.write(f"**Completeness:** {completeness:.1f}%")
                    
                    if col_type == "Numeric":
                        st.write(f"**Min:** {df[selected_column].min()}")
                        st.write(f"**Max:** {df[selected_column].max()}")
                        st.write(f"**Mean:** {df[selected_column].mean():.2f}")
                        st.write(f"**Median:** {df[selected_column].median()}")
                    elif col_type == "Text":
                        unique = df[selected_column].nunique()
                        st.write(f"**Unique Values:** {unique}")
            
            with explorer_col2:
                if selected_column:
                    if col_type == "Numeric":
                        fig = px.histogram(
                            df,
                            x=selected_column,
                            title=f"Distribution of {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif col_type == "Text":
                        value_counts = df[selected_column].value_counts().reset_index()
                        value_counts.columns = [selected_column, 'Count']
                        value_counts = value_counts.sort_values('Count', ascending=False).head(20)
                        
                        fig = px.bar(
                            value_counts,
                            x=selected_column,
                            y='Count',
                            title=f"Top Values for {selected_column}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Date
                        try:
                            date_series = pd.to_datetime(df[selected_column])
                            df_temp = pd.DataFrame({
                                'date': date_series,
                                'count': 1
                            })
                            df_temp['month'] = df_temp['date'].dt.strftime('%Y-%m')
                            
                            monthly_counts = df_temp.groupby('month')['count'].sum().reset_index()
                            
                            fig = px.line(
                                monthly_counts,
                                x='month',
                                y='count',
                                title=f"Timeline of {selected_column}",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not create timeline chart: {str(e)}")
            
            # Cross-column analysis
            st.subheader("Cross-Column Analysis")
            
            crosscol1, crosscol2 = st.columns(2)
            
            with crosscol1:
                x_col = st.selectbox("Select X-Axis Column", st.session_state.columns, key="xcol")
                y_col = st.selectbox("Select Y-Axis Column", 
                                     [c for c in st.session_state.numeric_columns if c != x_col],
                                     key="ycol")
                color_col = st.selectbox("Color By (Optional)", 
                                        ["None"] + [c for c in st.session_state.categorical_columns if c != x_col],
                                        key="colorcol")
                color_col = None if color_col == "None" else color_col
                
            with crosscol2:
                try:
                    if x_col in st.session_state.categorical_columns:
                        # For categorical x, use bar chart
                        fig = px.bar(
                            df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            title=f"{y_col} by {x_col}",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif x_col in st.session_state.date_columns:
                        # For date x, use line chart
                        try:
                            # Convert to datetime and group by month
                            df['date_temp'] = pd.to_datetime(df[x_col])
                            df['month_temp'] = df['date_temp'].dt.strftime('%Y-%m')
                            
                            # Group by month
                            if color_col:
                                monthly_data = df.groupby(['month_temp', color_col])[y_col].mean().reset_index()
                                fig = px.line(
                                    monthly_data,
                                    x='month_temp',
                                    y=y_col,
                                    color=color_col,
                                    title=f"{y_col} by Month",
                                    labels={"month_temp": "Month"},
                                    markers=True
                                )
                            else:
                                monthly_data = df.groupby('month_temp')[y_col].mean().reset_index()
                                fig = px.line(
                                    monthly_data,
                                    x='month_temp',
                                    y=y_col,
                                    title=f"{y_col} by Month",
                                    labels={"month_temp": "Month"},
                                    markers=True
                                )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not create time series chart: {str(e)}")
                    else:
                        # For numeric x, use scatter
                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            color=color_col,
                            title=f"{y_col} vs. {x_col}",
                            opacity=0.7,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create cross-column chart: {str(e)}")
                    
            # Data table with search
            st.subheader("Data Table")
            
            search_term = st.text_input("Search in data", "")
            
            if search_term:
                # Search across all columns
                mask = pd.Series(False, index=df.index)
                for col in df.columns:
                    if df[col].dtype == 'object':  # For string columns
                        mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
                    else:  # For numeric columns
                        # Convert search term to number if possible
                        try:
                            num_search = float(search_term)
                            mask = mask | (df[col] == num_search)
                        except:
                            pass
                
                search_results = df[mask]
                st.dataframe(search_results, use_container_width=True)
            else:
                # Show first 100 rows if no search
                st.dataframe(df.head(100), use_container_width=True)
                
        with tab3:
            st.header("Suggested Visualizations")
            
            df = st.session_state.filtered_data
            
            # Get the suggested visualizations from Claude
            suggested_vizs = st.session_state.suggested_visualizations
            
            if not suggested_vizs:
                st.info("No visualizations were suggested. Try analyzing the data again or create custom visualizations.")
            else:
                for i, viz in enumerate(suggested_vizs):
                    st.subheader(f"{i+1}. {viz.get('title', 'Visualization')}")
                    st.write(viz.get('description', 'No description provided.'))
                    
                    # Try to create a visualization based on the description
                    # This is a simplified implementation that creates basic charts
                    
                    col_map = st.session_state.llm_analysis.get('column_mapping', {})
                    
                    # Check if the description mentions specific columns
                    desc = viz.get('description', '').lower()
                    
                    # Prepare placeholder for the chart
                    chart_container = st.container()
                    
                    # Get a visualization from Claude
                    question = viz.get('description', '')
                    
                    if st.button(f"Generate Visualization #{i+1}"):
                        with st.spinner("Creating visualization..."):
                            recommendation = get_visualization_recommendation(df, question)
                            
                            if recommendation:
                                with chart_container:
                                    # Display the recommendation
                                    with st.expander("View recommendation details"):
                                        st.json(recommendation)
                                    
                                    # Create and display the visualization
                                    fig = create_visualization(df, recommendation)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Option to save chart
                                        if st.button(f"Save to Dashboard #{i+1}"):
                                            chart_entry = {
                                                "name": recommendation.get("title", viz.get('title', 'Untitled Chart')),
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "recommendation": recommendation,
                                                "query": question
                                            }
                                            st.session_state.chart_history.append(chart_entry)
                                            st.success("Chart saved to dashboard!")
                    
                    st.markdown("---")
                    
        with tab4:
            st.header("Custom Visualizations")
            
            df = st.session_state.filtered_data
            
            # Add a selector for visualization type
            viz_type = st.radio(
                "Choose visualization approach:",
                ["AI-Guided Visualization", "Manual Chart Builder"]
            )
            
            if viz_type == "AI-Guided Visualization":
                # LLM-guided visualization
                st.subheader("AI-Guided Visualization")
                
                question = st.text_input(
                    "What would you like to visualize? Describe in natural language",
                    placeholder="E.g., Show me the trend of billable hours over time by practice area"
                )
                
                if question and st.button("Generate Visualization"):
                    with st.spinner("Analyzing your data and generating visualization..."):
                        recommendation = get_visualization_recommendation(
                            df,
                            question
                        )
                        
                        if recommendation:
                            st.success("Visualization recommendation created!")
                            
                            # Display the recommendation
                            with st.expander("View recommendation details"):
                                st.json(recommendation)
                            
                            # Create and display the visualization
                            fig = create_visualization(df, recommendation)
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
            else:
                # Manual chart builder
                st.subheader("Manual Chart Builder")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    chart_type = st.selectbox(
                        "Chart Type",
                        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"]
                    )
                    
                    # Use all available columns
                    available_columns = st.session_state.columns
                    
                    x_axis = st.selectbox("X-Axis", available_columns)
                    
                    if chart_type != "Pie Chart":
                        y_axis = st.selectbox("Y-Axis", st.session_state.numeric_columns if st.session_state.numeric_columns else available_columns)
                    else:
                        y_axis = st.selectbox("Values", st.session_state.numeric_columns if st.session_state.numeric_columns else available_columns)
                    
                    color_by = st.selectbox("Color By (Optional)", ["None"] + st.session_state.categorical_columns)
                    color_by = None if color_by == "None" else color_by
                    
                    chart_title = st.text_input("Chart Title", f"{chart_type} of {y_axis} by {x_axis}")
                
                with chart_col2:
                    try:
                        if chart_type == "Bar Chart":
                            fig = px.bar(
                                df,
                                x=x_axis,
                                y=y_axis,
                                color=color_by,
                                title=chart_title
                            )
                        elif chart_type == "Line Chart":
                            fig = px.line(
                                df,
                                x=x_axis,
                                y=y_axis,
                                color=color_by,
                                title=chart_title
                            )
                        elif chart_type == "Scatter Plot":
                            fig = px.scatter(
                                df,
                                x=x_axis,
                                y=y_axis,
                                color=color_by,
                                title=chart_title
                            )
                        elif chart_type == "Pie Chart":
                            fig = px.pie(
                                df,
                                names=x_axis,
                                values=y_axis,
                                title=chart_title
                            )
                        elif chart_type == "Box Plot":
                            fig = px.box(
                                df,
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
else:
    # No data uploaded yet
    st.info("👈 Please upload a CSV file to get started.")
    
    # Sample data explanation
    st.markdown("""
    ## How to use this dashboard
    
    1. **Upload Data**: Start by uploading a CSV file with your law firm data.
    2. **AI Analysis**: Let Claude analyze your data and identify key legal fields.
    3. **Explore Visualizations**: View AI-suggested visualizations or create your own.
    4. **Filter & Customize**: Filter data and save visualizations to your dashboard.
    
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
    st.caption("v0.2.0")
