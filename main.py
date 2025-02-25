import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import numpy as np
from anthropic import Anthropic
import json
import time
import re

# Initialize session state variables
if "claude_api_key" not in st.session_state:
    st.session_state["claude_api_key"] = ""
if "claude_model" not in st.session_state:
    st.session_state["claude_model"] = "claude-3-haiku-20240307"
if "show_api_config" not in st.session_state:
    st.session_state["show_api_config"] = False
if "saved_visualizations" not in st.session_state:
    st.session_state["saved_visualizations"] = []

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
        # Convert date-related columns to datetime
        date_patterns = ['date', 'day', 'month', 'quarter', 'year']
        if any(pattern in col.lower() for pattern in date_patterns):
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

def categorize_columns(df):
    """Categorize columns into logical groups for easier selection."""
    columns = df.columns.tolist()
    categories = {
        'Time Tracking': [],
        'Financial': [],
        'Matter Info': [],
        'Attorney/User': [],
        'Contact/Client': [],
        'Time Period': [],
        'Other': []
    }
    
    for col in columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['hour', 'time', 'track', 'bill']):
            categories['Time Tracking'].append(col)
        elif any(term in col_lower for term in ['value', 'rate', 'cost', 'price', 'fee', 'revenue', 'profit', 'utilization']):
            categories['Financial'].append(col)
        elif any(term in col_lower for term in ['matter', 'practice area', 'case']):
            categories['Matter Info'].append(col)
        elif any(term in col_lower for term in ['attorney', 'user', 'responsible', 'originating']):
            categories['Attorney/User'].append(col)
        elif any(term in col_lower for term in ['contact', 'client', 'company']):
            categories['Contact/Client'].append(col)
        elif any(term in col_lower for term in ['date', 'day', 'month', 'quarter', 'year']):
            categories['Time Period'].append(col)
        else:
            categories['Other'].append(col)
    
    return categories

def setup_claude_client():
    """Set up and return the Claude API client using stored credentials."""
    api_key = st.session_state.get("claude_api_key", "")
    if not api_key:
        return None
    
    return Anthropic(api_key=api_key)

def generate_visualization_from_nl_query(df, query, client, column_categories):
    """
    Use Claude to interpret a natural language query and generate a visualization.
    
    Args:
        df: The pandas DataFrame containing the data
        query: The natural language query string
        client: The Anthropic API client
        column_categories: Dictionary of columns grouped by category
        
    Returns:
        A dictionary with visualization details or an error message
    """
    if not client:
        return {"error": "Claude API key not set up. Please configure it in the settings."}
    
    # Get DataFrame info to provide context to Claude
    column_info = {}
    for col in df.columns:
        column_info[col] = {
            "dtype": str(df[col].dtype),
            "sample_values": df[col].sample(min(5, len(df))).tolist(),
            "num_unique": df[col].nunique(),
            "has_nulls": df[col].isnull().any()
        }
    
    # Add column category information
    column_categories_flat = {}
    for category, cols in column_categories.items():
        for col in cols:
            column_categories_flat[col] = category
    
    # Craft prompt for Claude
    prompt = f"""
    You are an analytics assistant for a law firm that uses Clio practice management software.
    The data includes fields related to time tracking, billable hours, matters (cases), attorneys, clients, and more.
    
    The user has uploaded a dataframe with the following columns, categorized by type:
    {json.dumps(column_categories, indent=2)}
    
    Here's detailed information about each column:
    {json.dumps(column_info, indent=2)}
    
    The user is asking: "{query}"
    
    Based on this query, determine:
    1. The most appropriate visualization type (bar, line, pie, scatter, etc.)
    2. Which columns to use for x-axis/category
    3. Which columns to use for y-axis/values
    4. Any filters to apply
    5. Any aggregation to apply (sum, mean, count, etc.)
    6. A title for the visualization
    
    Return ONLY a JSON object with the following structure:
    {{
        "viz_type": "bar|line|pie|scatter|heatmap",
        "x_column": "column_name",
        "y_column": "column_name",
        "color_column": "column_name" (optional),
        "filters": [
            {{"column": "column_name", "operation": "==|>|<|>=|<=|in", "value": value}}
        ],
        "aggregation": "sum|mean|count|median|min|max",
        "title": "Suggested title for the visualization"
    }}
    
    Do not include any explanation or additional text. Only return a valid JSON object that can be parsed.
    """
    
    try:
        response = client.messages.create(
            model=st.session_state["claude_model"],
            max_tokens=1000,
            temperature=0,
            system="You are a data visualization assistant for a law firm that converts natural language requests into JSON visualization specifications.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        output = response.content[0].text
        
        # Try to parse the JSON response
        try:
            viz_spec = json.loads(output)
            return viz_spec
        except json.JSONDecodeError:
            # If we can't parse the JSON directly, try to extract it from the text
            import re
            json_match = re.search(r'({.*})', output.replace('\n', ' '), re.DOTALL)
            if json_match:
                try:
                    viz_spec = json.loads(json_match.group(1))
                    return viz_spec
                except json.JSONDecodeError:
                    return {"error": "Failed to parse Claude's response as JSON."}
            else:
                return {"error": "Failed to extract JSON from Claude's response."}
        
    except Exception as e:
        return {"error": f"Error calling Claude API: {str(e)}"}

def create_visualization_from_spec(df, viz_spec):
    """
    Create and return a Plotly visualization based on the specification.
    
    Args:
        df: The pandas DataFrame
        viz_spec: The visualization specification dict from Claude
        
    Returns:
        A Plotly figure or error message
    """
    try:
        # Apply filters if any
        filtered_df = df.copy()
        if "filters" in viz_spec and viz_spec["filters"]:
            for filter_spec in viz_spec["filters"]:
                col = filter_spec["column"]
                op = filter_spec["operation"]
                val = filter_spec["value"]
                
                if op == "==":
                    filtered_df = filtered_df[filtered_df[col] == val]
                elif op == ">":
                    filtered_df = filtered_df[filtered_df[col] > val]
                elif op == "<":
                    filtered_df = filtered_df[filtered_df[col] < val]
                elif op == ">=":
                    filtered_df = filtered_df[filtered_df[col] >= val]
                elif op == "<=":
                    filtered_df = filtered_df[filtered_df[col] <= val]
                elif op == "in" and isinstance(val, list):
                    filtered_df = filtered_df[filtered_df[col].isin(val)]
        
        # Apply aggregation if specified
        x_col = viz_spec["x_column"]
        y_col = viz_spec["y_column"]
        
        if "aggregation" in viz_spec and viz_spec["aggregation"]:
            agg_func = viz_spec["aggregation"]
            agg_df = filtered_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
        else:
            agg_df = filtered_df
        
        # Create the visualization based on type
        viz_type = viz_spec["viz_type"].lower()
        title = viz_spec.get("title", f"{y_col} by {x_col}")
        
        color_col = viz_spec.get("color_column")
        
        if viz_type == "bar":
            if color_col and color_col in filtered_df.columns:
                fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                fig = px.bar(agg_df, x=x_col, y=y_col, title=title)
        
        elif viz_type == "line":
            fig = px.line(agg_df, x=x_col, y=y_col, title=title, markers=True)
            
        elif viz_type == "pie":
            fig = px.pie(agg_df, names=x_col, values=y_col, title=title)
            
        elif viz_type == "scatter":
            if color_col and color_col in filtered_df.columns:
                fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col, title=title)
            else:
                fig = px.scatter(filtered_df, x=x_col, y=y_col, title=title)
                
        elif viz_type == "heatmap":
            # For heatmap, we need a pivot table
            pivot_df = filtered_df.pivot_table(
                index=x_col, 
                columns=color_col if color_col else viz_spec.get("columns_column"), 
                values=y_col,
                aggfunc=viz_spec.get("aggregation", "mean")
            )
            fig = px.imshow(
                pivot_df, 
                title=title,
                color_continuous_scale="blues"
            )
            
        else:
            return {"error": f"Unsupported visualization type: {viz_type}"}
        
        return {"figure": fig}
        
    except Exception as e:
        return {"error": f"Error creating visualization: {str(e)}"}

def render_nl_query_section(df, column_categories):
    st.subheader("Ask for a Specific Visualization")
    
    # Check if Claude API key is set
    if "claude_api_key" not in st.session_state or not st.session_state["claude_api_key"]:
        st.warning("Claude API key not set. Please configure it in the sidebar settings.")
        if st.button("Configure Claude API"):
            st.session_state["show_api_config"] = True
        return
    
    # Input for natural language query
    nl_query = st.text_input(
        "Describe the visualization you want",
        placeholder="E.g., Show me billable hours by attorney over time"
    )
    
    # Only show the generate button if there's a query
    if nl_query:
        generate_col1, generate_col2 = st.columns([1, 3])
        
        with generate_col1:
            generate_button = st.button("Generate Visualization")
        
        with generate_col2:
            # Dynamic example queries based on available columns
            examples = []
            
            if column_categories['Time Tracking'] and column_categories['Attorney/User']:
                examples.append(f"Show me total {column_categories['Time Tracking'][0]} by {column_categories['Attorney/User'][0]}")
            
            if column_categories['Financial'] and column_categories['Time Period']:
                examples.append(f"What's the trend of {column_categories['Financial'][0]} over {column_categories['Time Period'][0]}?")
            
            if column_categories['Time Tracking'] and len(column_categories['Time Tracking']) > 1:
                examples.append(f"Compare {column_categories['Time Tracking'][0]} vs {column_categories['Time Tracking'][1]}")
            
            if column_categories['Matter Info'] and column_categories['Financial']:
                examples.append(f"Which {column_categories['Matter Info'][0]} have the highest {column_categories['Financial'][0]}?")
            
            if len(examples) == 0:
                examples = [
                    "Show me total billable hours by attorney",
                    "What's the trend of utilization rate over time?",
                    "Compare billable vs non-billable hours",
                    "Which matters have the highest revenue?"
                ]
                
            st.session_state["examples"] = examples
            
            selected_example = st.selectbox(
                "Or try an example query:",
                [""] + st.session_state["examples"]
            )
            
            if selected_example and selected_example != nl_query:
                nl_query = selected_example
                st.experimental_rerun()
        
        if generate_button and nl_query:
            with st.spinner("Generating visualization with Claude..."):
                # Get the Claude client
                client = setup_claude_client()
                
                # Generate visualization spec
                viz_spec = generate_visualization_from_nl_query(df, nl_query, client, column_categories)
                
                if "error" in viz_spec:
                    st.error(viz_spec["error"])
                else:
                    # Store the spec in session state for inspection
                    st.session_state["last_viz_spec"] = viz_spec
                    
                    # Show the generated spec in an expander
                    with st.expander("View visualization specification"):
                        st.json(viz_spec)
                    
                    # Create the visualization
                    viz_result = create_visualization_from_spec(df, viz_spec)
                    
                    if "error" in viz_result:
                        st.error(viz_result["error"])
                    else:
                        st.plotly_chart(viz_result["figure"], use_container_width=True)
                        
                        # Option to add this to dashboard
                        if st.button("Add to Dashboard"):
                            if "saved_visualizations" not in st.session_state:
                                st.session_state["saved_visualizations"] = []
                            
                            st.session_state["saved_visualizations"].append({
                                "query": nl_query,
                                "spec": viz_spec,
                                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            st.success("Visualization added to your dashboard!")

def render_claude_config_sidebar():
    st.sidebar.header("Claude API Configuration")
    
    # Get existing API key from session state
    api_key = st.session_state.get("claude_api_key", "")
    
    # Add API key input
    new_api_key = st.sidebar.text_input(
        "Claude API Key",
        value=api_key,
        type="password",
        help="Enter your Anthropic Claude API key"
    )
    
    # Save API key to session state if changed
    if new_api_key != api_key:
        st.session_state["claude_api_key"] = new_api_key
        
        # Test the API key if it's been provided
        if new_api_key:
            with st.sidebar.spinner("Testing API key..."):
                client = setup_claude_client()
                try:
                    # Simple test message
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=10,
                        messages=[
                            {"role": "user", "content": "Respond with 'OK' only."}
                        ]
                    )
                    if "OK" in response.content[0].text:
                        st.sidebar.success("API key is valid!")
                    else:
                        st.sidebar.warning("API key might be valid, but received unexpected response.")
                except Exception as e:
                    st.sidebar.error(f"Error testing API key: {str(e)}")
    
    # Add model selection
    models = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    selected_model = st.sidebar.selectbox(
        "Claude Model",
        models,
        index=0,
        help="Select which Claude model to use. Haiku is fastest, Opus is most capable."
    )
    
    # Save selected model to session state
    st.session_state["claude_model"] = selected_model

def detect_billing_fields(df):
    """Detect available billing/hour fields in the dataframe."""
    billing_fields = {
        'billable_hours': None,
        'non_billable_hours': None,
        'total_hours': None,
        'billable_value': None,
        'utilization_rate': None
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if 'billed hours' in col_lower or 'billable hours' in col_lower:
            billing_fields['billable_hours'] = col
        elif 'non-billable hours' in col_lower or 'non billable hours' in col_lower:
            billing_fields['non_billable_hours'] = col
        elif 'tracked hours' in col_lower or 'total hours' in col_lower:
            billing_fields['total_hours'] = col
        elif 'billed hours value' in col_lower or 'billable hours value' in col_lower:
            billing_fields['billable_value'] = col
        elif 'utilization rate' in col_lower:
            billing_fields['utilization_rate'] = col
    
    return billing_fields

def detect_entity_fields(df):
    """Detect available entity fields (attorney, matter, client) in the dataframe."""
    entity_fields = {
        'attorney': None,
        'matter': None,
        'practice_area': None,
        'client': None,
        'date': None
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if 'attorney' in col_lower and 'name' in col_lower:
            entity_fields['attorney'] = col
        elif 'matter description' in col_lower:
            entity_fields['matter'] = col
        elif 'practice area' in col_lower:
            entity_fields['practice_area'] = col
        elif ('client' in col_lower or 'company name' in col_lower) and 'name' in col_lower:
            entity_fields['client'] = col
        elif 'activity date' in col_lower or 'date' in col_lower:
            entity_fields['date'] = col
    
    return entity_fields

def generate_default_visualizations(df, billing_fields, entity_fields):
    """Generate default visualizations based on available fields."""
    visualizations = []
    
    # 1. Billable vs Non-Billable Hours
    if billing_fields['billable_hours'] and billing_fields['non_billable_hours']:
        billed_vs_unbilled = pd.DataFrame({
            'Category': ['Billable Hours', 'Non-Billable Hours'],
            'Hours': [
                df[billing_fields['billable_hours']].sum(), 
                df[billing_fields['non_billable_hours']].sum()
            ]
        })
        
        fig1 = px.pie(
            billed_vs_unbilled,
            values='Hours',
            names='Category',
            title='Billable vs Non-Billable Hours',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        visualizations.append(("pie_billable_split", fig1))
    
    # 2. Utilization Rate by Attorney
    if billing_fields['utilization_rate'] and entity_fields['attorney']:
        util_by_atty = df.groupby(entity_fields['attorney'])[billing_fields['utilization_rate']].mean().reset_index()
        util_by_atty = util_by_atty.sort_values(billing_fields['utilization_rate'], ascending=False)
        
        fig2 = px.bar(
            util_by_atty, 
            x=entity_fields['attorney'], 
            y=billing_fields['utilization_rate'],
            title='Average Utilization Rate by Attorney',
            color=billing_fields['utilization_rate'],
            color_continuous_scale='blues'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        visualizations.append(("bar_utilization", fig2))
    
    # 3. Practice Area Distribution
    if entity_fields['practice_area']:
        practice_dist = df[entity_fields['practice_area']].value_counts().reset_index()
        practice_dist.columns = ['Practice Area', 'Count']
        
        fig3 = px.bar(
            practice_dist,
            x='Practice Area',
            y='Count',
            title='Matter Distribution by Practice Area',
            color='Count',
            color_continuous_scale='blues'
        )
        fig3.update_layout(xaxis_tickangle=-45)
        visualizations.append(("bar_practice", fig3))
    
    # 4. Hours Over Time
    time_col = None
    for col in df.columns:
        if 'activity month' in col.lower() or 'month' in col.lower():
            time_col = col
            break
    
    if time_col and billing_fields['billable_hours']:
        monthly_billed = df.groupby(time_col)[billing_fields['billable_hours']].sum().reset_index()
        
        fig4 = px.line(
            monthly_billed,
            x=time_col,
            y=billing_fields['billable_hours'],
            title='Monthly Trend of Billable Hours',
            markers=True
        )
        visualizations.append(("line_monthly", fig4))
    
    return visualizations

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
    
    # Categorize columns
    column_categories = categorize_columns(df)
    
    # Detect important fields
    billing_fields = detect_billing_fields(df)
    entity_fields = detect_entity_fields(df)
    
    # Ask about filters
    st.sidebar.header("Filter Options")
    st.sidebar.info("Select the filters you want to include in your dashboard")
    
    # Create filter options based on available columns
    filter_options = {}
    
    # Date range filter if date columns exist
    date_cols = column_categories['Time Period']
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
    attorney_cols = [col for col in df.columns if 'attorney' in col.lower() and 'name' in col.lower()]
    if attorney_cols:
        show_attorney_filter = st.sidebar.checkbox("Attorney Filter", value=True)
        if show_attorney_filter:
            selected_attorney_col = attorney_cols[0]  # Use the first attorney column
            attorneys = ['All'] + sorted(df[selected_attorney_col].unique().tolist())
            selected_attorneys = st.sidebar.multiselect("Select Attorneys", attorneys, default=['All'])
            if 'All' not in selected_attorneys:
                df = df[df[selected_attorney_col].isin(selected_attorneys)]
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
    billable_cols = [col for col in df.columns if 'billable matter' in col.lower() or 'billable' in col.lower()]
    if billable_cols:
        show_billable_filter = st.sidebar.checkbox("Billable Matters Only", value=False)
        if show_billable_filter:
            billable_col = billable_cols[0]
            if df[billable_col].dtype == np.dtype('bool'):
                df = df[df[billable_col] == True]
            else:
                try:
                    df = df[df[billable_col] == 1]
                except:
                    st.sidebar.warning(f"Unable to filter by {billable_col}. Check the data format.")
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
    
    overview_cols = st.columns(3)
    with overview_cols[0]:
        st.metric("Total Records", f"{len(df):,}")
    with overview_cols[1]:
        st.metric("Total Columns", len(df.columns))
    with overview_cols[2]:
        date_range = ""
        if date_cols and len(date_cols) > 0:
            try:
                min_date = df[date_cols[0]].min().strftime('%Y-%m-%d')
                max_date = df[date_cols[0]].max().strftime('%Y-%m-%d')
                date_range = f"{min_date} to {max_date}"
            except:
                date_range = "Unknown"
        st.metric("Date Range", date_range)
    
    # Allow users to view the raw data if they want
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    # Display column information
    with st.expander("Column Categories"):
        for category, cols in column_categories.items():
            if cols:  # Only show categories that have columns
                st.subheader(category)
                st.write(", ".join(cols))
    
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
        add_data_tabs = st.tabs(["Upload Additional File", "Connect to External Source", "Manual Entry"])
        
        with add_data_tabs[0]:
            st.subheader("Upload Additional Data")
            additional_file = st.file_uploader("Upload additional CSV file", type="csv", key="additional_file")
            merge_option = st.selectbox(
                "How should the data be merged?",
                ["Append (add rows)", "Join on common column", "Replace current data"]
            )
            if additional_file and st.button("Process Additional File"):
                st.info("This would merge the additional data with your current dataset. (Demo mode)")
        
        with add_data_tabs[1]:
            st.subheader("Connect to External Source")
            data_source = st.selectbox(
                "Select external data source",
                ["Clio API", "QuickBooks", "SQL Database", "Excel/CSV from network location"]
            )
            st.text_input("Connection string/URL")
            st.text_input("Authentication credentials", type="password")
            if st.button("Test Connection"):
                st.info("This would connect to your external data source. (Demo mode)")
        
        with add_data_tabs[2]:
            st.subheader("Manual Data Entry")
            st.info("This section would allow manual entry or correction of data records. (Demo mode)")
            st.text_area("Enter data in CSV format or use the table editor below")
            if st.button("Add Manual Data"):
                st.info("This would add your manually entered data. (Demo mode)")
    
    # Configure Claude API in the sidebar
    st.sidebar.markdown("---")
    render_claude_config_sidebar()
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Default Visualizations", "Custom Analysis", "Attorney Performance", "Matter Insights", "Real-Time Connection"])
    
    with tab1:
        st.header("Default Visualizations")
        
        # Generate default visualizations based on available fields
        default_visualizations = generate_default_visualizations(df, billing_fields, entity_fields)
        
        if not default_visualizations:
            st.warning("Not enough data to generate default visualizations. Please upload a file with more Clio fields.")
        else:
            # Show visualizations in a 2-column layout
            for i in range(0, len(default_visualizations), 2):
                cols = st.columns(2)
                
                with cols[0]:
                    if i < len(default_visualizations):
                        _, fig = default_visualizations[i]
                        st.plotly_chart(fig, use_container_width=True)
                
                with cols[1]:
                    if i + 1 < len(default_visualizations):
                        _, fig = default_visualizations[i + 1]
                        st.plotly_chart(fig, use_container_width=True)
        
        # Additional dynamic visualizations based on available data
        if billing_fields['billable_hours'] and entity_fields['matter'] and len(df) > 0:
            st.subheader("Top Matters by Billable Hours")
            
            top_matters = df.groupby(entity_fields['matter'])[billing_fields['billable_hours']].sum().reset_index()
            top_matters = top_matters.sort_values(billing_fields['billable_hours'], ascending=False).head(10)
            
            fig = px.bar(
                top_matters,
                x=billing_fields['billable_hours'],
                y=entity_fields['matter'],
                orientation='h',
                title='Top 10 Matters by Billable Hours',
                color=billing_fields['billable_hours'],
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if billing_fields['utilization_rate'] and len(df) > 0:
            st.subheader("Utilization Rate Distribution")
            
            fig = px.histogram(
                df,
                x=billing_fields['utilization_rate'],
                nbins=20,
                title='Distribution of Utilization Rates',
                color_discrete_sequence=['#4e8df5']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Custom Analysis")
        
        # Add tabs within this section
        custom_tabs = st.tabs(["Custom Visualization Builder", "Natural Language Query", "Saved Visualizations"])
        
        with custom_tabs[0]:
            # Allow users to select dimensions for custom visualization
            st.subheader("Create Your Own Visualization")
            
            viz_type = st.selectbox(
                "Select Visualization Type", 
                ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Heatmap"]
            )
            
            # Use column categories for more logical selection
            if viz_type in ["Bar Chart", "Line Chart"]:
                # Get all column categories in a flat list for selection dropdowns
                categorical_cols = []
                for category, cols in column_categories.items():
                    categorical_cols.extend(cols)
                
                numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                
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
                # Offer categorized column selection
                category_options = {
                    "Attorney": [col for col in df.columns if 'attorney' in col.lower()],
                    "Matter": [col for col in df.columns if 'matter' in col.lower()],
                    "Practice Area": [col for col in df.columns if 'practice' in col.lower()],
                    "Client": [col for col in df.columns if 'client' in col.lower() or 'company' in col.lower()],
                    "Other": [col for col in df.columns if not any(term in col.lower() for term in ['attorney', 'matter', 'practice', 'client', 'company'])]
                }
                
                # Flatten the options and remove empty categories
                flat_options = []
                for cat, cols in category_options.items():
                    if cols:
                        flat_options.extend(cols)
                
                category = st.selectbox("Select Category for Segments", flat_options)
                
                # Values to use for pie size
                value_options = {
                    "Hours": [col for col in df.columns if 'hour' in col.lower()],
                    "Financial": [col for col in df.columns if any(term in col.lower() for term in ['value', 'cost', 'rate', 'revenue'])],
                    "Counts": ["Count (number of records)"]
                }
                
                # Flatten value options
                flat_value_options = []
                for cat, cols in value_options.items():
                    if cols:
                        flat_value_options.extend(cols)
                
                value = st.selectbox("Select Value for Size", flat_value_options)
                
                if st.button("Generate Visualization"):
                    # Handle count option specially
                    if value == "Count (number of records)":
                        pie_data = df[category].value_counts().reset_index()
                        pie_data.columns = [category, 'Count']
                        
                        fig = px.pie(
                            pie_data,
                            values='Count',
                            names=category,
                            title=f"Distribution of {category}"
                        )
                    else:
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
                numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
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
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                
                row_cat = st.selectbox("Select Row Category", categorical_cols, key="heatmap_row")
                col_cat = st.selectbox("Select Column Category", categorical_cols, key="heatmap_col")
                value_col = st.selectbox("Select Value", numeric_cols, key="heatmap_val")
                agg_func = st.selectbox("Select Aggregation Function", ["Mean", "Sum", "Count", "Median"], key="heatmap_agg")
                
                # Map to pandas function
                agg_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Count": "count",
                    "Median": "median"
                }
                
                if st.button("Generate Visualization"):
                    # Create a pivot table for the heatmap
                    try:
                        heatmap_data = df.pivot_table(
                            index=row_cat,
                            columns=col_cat,
                            values=value_col,
                            aggfunc=agg_map[agg_func]
                        ).fillna(0)
                        
                        fig = px.imshow(
                            heatmap_data,
                            title=f"Heatmap of {value_col} by {row_cat} and {col_cat} ({agg_func})",
                            color_continuous_scale='blues'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating heatmap: {str(e)}")
                        st.info("Tip: For heatmaps, choose categories with a manageable number of unique values.")
        
        with custom_tabs[1]:
            # This is where we add the Claude-powered NL query feature
            render_nl_query_section(df, column_categories)
        
        with custom_tabs[2]:
            # Show saved visualizations
            st.subheader("Your Saved Visualizations")
            
            if "saved_visualizations" in st.session_state and st.session_state["saved_visualizations"]:
                for i, viz in enumerate(st.session_state["saved_visualizations"]):
                    with st.expander(f"{viz['query']} (saved {viz['timestamp']})"):
                        # Recreate visualization from saved spec
                        viz_result = create_visualization_from_spec(df, viz["spec"])
                        
                        if "error" in viz_result:
                            st.error(viz_result["error"])
                        else:
                            st.plotly_chart(viz_result["figure"], use_container_width=True)
                            
                            # Option to remove this visualization
                            if st.button(f"Remove", key=f"remove_{i}"):
                                st.session_state["saved_visualizations"].pop(i)
                                st.experimental_rerun()
            else:
                st.info("No saved visualizations yet. Create some using the Natural Language Query tab!")
    
    with tab3:
        st.header("Attorney Performance Dashboard")
        
        # Check if required columns exist
        attorney_cols = [col for col in df.columns if 'attorney' in col.lower() and 'name' in col.lower()]
        if attorney_cols:
            attorney_col = attorney_cols[0]  # Use the first attorney column
            
            # Filter for specific attorney
            attorneys = sorted(df[attorney_col].unique())
            selected_attorney = st.selectbox("Select Attorney", attorneys)
            
            # Filter data for selected attorney
            atty_data = df[df[attorney_col] == selected_attorney]
            
            # Top metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                if billing_fields['billable_hours']:
                    total_billed = atty_data[billing_fields['billable_hours']].sum()
                    st.metric("Total Billed Hours", f"{total_billed:.1f}")
            
            with metric_cols[1]:
                if billing_fields['utilization_rate']:
                    avg_util = atty_data[billing_fields['utilization_rate']].mean()
                    st.metric("Avg Utilization Rate", f"{avg_util:.1f}%")
            
            with metric_cols[2]:
                if billing_fields['billable_value']:
                    total_value = atty_data[billing_fields['billable_value']].sum()
                    st.metric("Total Billed Value", f"${total_value:,.2f}")
            
            with metric_cols[3]:
                if entity_fields['matter']:
                    matter_count = atty_data[entity_fields['matter']].nunique()
                    st.metric("Total Matters", matter_count)
            
            # Performance charts
            perf_cols = st.columns(2)
            
            with perf_cols[0]:
                if billing_fields['billable_hours'] and 'Activity month' in df.columns:
                    monthly_billed = atty_data.groupby('Activity month')[billing_fields['billable_hours']].sum().reset_index()
                    
                    fig = px.line(
                        monthly_billed,
                        x='Activity month',
                        y=billing_fields['billable_hours'],
                        title=f'Monthly Billed Hours - {selected_attorney}',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with perf_cols[1]:
                if entity_fields['practice_area'] and billing_fields['billable_hours']:
                    practice_hours = atty_data.groupby(entity_fields['practice_area'])[billing_fields['billable_hours']].sum().reset_index()
                    practice_hours = practice_hours.sort_values(billing_fields['billable_hours'], ascending=False)
                    
                    fig = px.pie(
                        practice_hours,
                        values=billing_fields['billable_hours'],
                        names=entity_fields['practice_area'],
                        title=f'Hours by Practice Area - {selected_attorney}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Matter table
            if entity_fields['matter']:
                st.subheader(f"Matters Handled by {selected_attorney}")
                
                matter_summary_dict = {
                    entity_fields['matter']: 'first'
                }
                
                if billing_fields['billable_hours']:
                    matter_summary_dict[billing_fields['billable_hours']] = 'sum'
                
                if billing_fields['billable_value']:
                    matter_summary_dict[billing_fields['billable_value']] = 'sum'
                
                if entity_fields['practice_area']:
                    matter_summary_dict[entity_fields['practice_area']] = 'first'
                
                matter_summary = atty_data.groupby(entity_fields['matter']).agg(matter_summary_dict).reset_index()
                
                if billing_fields['billable_hours']:
                    matter_summary = matter_summary.sort_values(billing_fields['billable_hours'], ascending=False)
                
                st.dataframe(matter_summary)
        else:
            st.warning("Attorney data is missing from the uploaded file.")
    
    with tab4:
        st.header("Matter Insights")
        
        # Check if required columns exist
        if entity_fields['matter']:
            # Filter for specific matter
            matters = sorted(df[entity_fields['matter']].unique())
            selected_matter = st.selectbox("Select Matter", matters)
            
            # Filter data for selected matter
            matter_data = df[df[entity_fields['matter']] == selected_matter]
            
            # Get matter metadata
            matter_info_cols = st.columns(4)
            
            with matter_info_cols[0]:
                if entity_fields['practice_area']:
                    practice = matter_data[entity_fields['practice_area']].iloc[0] if not matter_data[entity_fields['practice_area']].empty else "Unknown"
                    st.metric("Practice Area", practice)
            
            with matter_info_cols[1]:
                status_col = next((col for col in df.columns if 'matter status' in col.lower()), None)
                if status_col:
                    status = matter_data[status_col].iloc[0] if not matter_data[status_col].empty else "Unknown"
                    st.metric("Status", status)
            
            with matter_info_cols[2]:
                resp_atty_col = next((col for col in df.columns if 'responsible attorney' in col.lower()), None)
                if resp_atty_col:
                    resp_atty = matter_data[resp_atty_col].iloc[0] if not matter_data[resp_atty_col].empty else "Unknown"
                    st.metric("Responsible Attorney", resp_atty)
            
            with matter_info_cols[3]:
                if billing_fields['billable_hours']:
                    total_hours = matter_data[billing_fields['billable_hours']].sum()
                    st.metric("Total Hours", f"{total_hours:.1f}")
            
            # Matter analytics
            matter_cols = st.columns(2)
            
            with matter_cols[0]:
                time_col = next((col for col in df.columns if 'activity month' in col.lower() or 'month' in col.lower()), None)
                if billing_fields['billable_hours'] and time_col:
                    monthly_matter = matter_data.groupby(time_col)[billing_fields['billable_hours']].sum().reset_index()
                    
                    fig = px.bar(
                        monthly_matter,
                        x=time_col,
                        y=billing_fields['billable_hours'],
                        title=f'Monthly Hours - {selected_matter}',
                        color=billing_fields['billable_hours'],
                        color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with matter_cols[1]:
                if attorney_cols and billing_fields['billable_hours']:
                    atty_contrib = matter_data.groupby(attorney_cols[0])[billing_fields['billable_hours']].sum().reset_index()
                    atty_contrib = atty_contrib.sort_values(billing_fields['billable_hours'], ascending=False)
                    
                    fig = px.pie(
                        atty_contrib,
                        values=billing_fields['billable_hours'],
                        names=attorney_cols[0],
                        title=f'Attorney Contribution - {selected_matter}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Billable vs non-billable breakdown
            if billing_fields['billable_hours'] and billing_fields['non_billable_hours']:
                bill_cols = st.columns(2)
                
                with bill_cols[0]:
                    bill_breakdown = pd.DataFrame({
                        'Category': ['Billable', 'Non-Billable'],
                        'Hours': [
                            matter_data[billing_fields['billable_hours']].sum(),
                            matter_data[billing_fields['non_billable_hours']].sum()
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
                    if billing_fields['billable_value'] and 'Non-billable hours value' in df.columns:
                        value_breakdown = pd.DataFrame({
                            'Category': ['Billable', 'Non-Billable'],
                            'Value': [
                                matter_data[billing_fields['billable_value']].sum(),
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

    # Add a feedback section
    st.markdown("---")
    st.subheader("Dashboard Feedback")
    
    feedback = st.text_area("Please provide any feedback on this dashboard prototype:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! In a production version, this would be saved.")

else:
    # Instructions and sample visualization when no file is uploaded
    st.info("Please upload a CSV file exported from Clio to generate the analytics dashboard.")
    
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

# Footer
st.markdown("---")
st.markdown("© 2025 Law Firm Analytics Dashboard | Prototype Version")
