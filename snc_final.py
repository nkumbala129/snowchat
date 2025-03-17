# Ensure PyYAML is installed.
try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyYAML is not installed")

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import requests
import snowflake.connector
import tempfile, os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple


# Snowflake/Cortex Configuration
HOST = "GNB14769.snowflakecomputing.com"
DATABASE = "CORTEX_SEARCH_TUTORIAL_DB"
SCHEMA = "PUBLIC"
STAGE = "CC_STAGE"
FILE = "Climate_Career_Final_SM_Draft.yaml"

# Streamlit App Title
#st.title("Cortex Analyst")
#st.markdown(f"Semantic Model: `{FILE}`")
if "title_rendered" not in st.session_state:
    st.title("Cortex Analyst")
    st.markdown(f"Semantic Model: `{FILE}`")
    st.session_state.title_rendered = True


# User Authentication
if "username" not in st.session_state or "password" not in st.session_state:
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = st.session_state.get("CONN") or snowflake.connector.connect(  # Using shared connection
                user=st.session_state.username,
                password=st.session_state.password,
                account="GNB14769",
                host=HOST,
                port=443,
                warehouse="CORTEX_SEARCH_TUTORIAL_WH",
                role="DEV_BR_CORTEX_AI_ROLE",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            st.session_state.authenticated = True
            st.success("Authentication successful!")
            st.rerun()  # Refresh to load chat UI
        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:

    def get_or_init_snowflake_conn():
        """Retrieves the Snowflake connection from session state or initializes a new one."""
        if "CONN" not in st.session_state or st.session_state.CONN is None:
            try:
                conn = st.session_state.get("CONN") or snowflake.connector.connect(  # Using shared connection
                    user=st.session_state.username,
                    password=st.session_state.password,
                    account="GNB14769",
                    host=HOST,
                    port=443,
                    warehouse="CORTEX_SEARCH_TUTORIAL_WH",
                    role="DEV_BR_CORTEX_AI_ROLE",
                    database=DATABASE,
                    schema=SCHEMA,
                )
                st.session_state.CONN = conn
                return conn
            except Exception as e:
                st.error(f"Failed to connect to Snowflake: {e}")
                return None
        return st.session_state.CONN


###############################################################################
# Function to Load the Semantic Model from Snowflake Stage
###############################################################################
    def load_semantic_model() -> dict:
        """
        Downloads the YAML semantic model file from the Snowflake stage into a temporary
        directory and loads it using PyYAML.
        """
        conn = get_or_init_snowflake_conn()
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, FILE)
        try:
            cur = conn.cursor()
            # Download the file from the stage.
            get_sql = f"GET @{DATABASE}.{SCHEMA}.{STAGE}/{FILE} file://{temp_dir}"
            cur.execute(get_sql)
            cur.close()
            if not os.path.exists(local_path):
                st.error(f"Downloaded semantic model file not found at {local_path}")
                return {}
            with open(local_path, "r") as f:
                semantic_model = yaml.safe_load(f)
            return semantic_model
        except Exception as e:
            st.error(f"Error loading semantic model from Snowflake stage: {e}")
            return {}
    
    ###############################################################################
    # Helper Functions to Extract Table and Relationship Info
    ###############################################################################
    def get_table_info(semantic_model: dict, table_name: str) -> dict:
        """
        Searches semantic_model["tables"] for the table with the given name.
        Returns its dictionary (including base_table, measures, dimensions, etc.).
        """
        for t in semantic_model.get("tables", []):
            if t.get("name") == table_name:
                return t
        return {}
    
    def get_relationship(semantic_model: dict, left_table: str, right_table: str) -> Optional[dict]:
        """
        Searches semantic_model["relationships"] for a relationship where left_table equals
        left_table and right_table equals right_table.
        Returns the relationship dict if found.
        """
        for rel in semantic_model.get("relationships", []):
            if rel.get("left_table") == left_table and rel.get("right_table") == right_table:
                return rel
        return None

    
    def send_message(prompt: str, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Calls the Cortex Analyst REST API and returns the response.
        (This service is assumed to read the semantic model and generate the SQL
        with correct JOINs and filters based on the natural language prompt.)
        """
        request_body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
        }
        # If no token is provided, get the connection and its token.
        if token is None:
            conn = get_or_init_snowflake_conn()
            token = conn.rest.token
        resp = requests.post(
            url=f"https://{HOST}/api/v2/cortex/analyst/message",
            json=request_body,
            headers={
                "Authorization": f'Snowflake Token="{token}"',
                "Content-Type": "application/json",
            },
            timeout=50
        )
        request_id = resp.headers.get("X-Snowflake-Request-Id", "N/A")
        if resp.status_code < 400:
            return {**resp.json(), "request_id": request_id}
        else:
            raise Exception(
                f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
            )
    
    ###############################################################################
    # Chat and Message Display Functions
    ###############################################################################
    def process_message(prompt: str) -> None:
        """Processes a chat message and updates session state (used by the Chat page)."""
        st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = send_message(prompt=prompt)
                request_id = response["request_id"]
                content = response["message"]["content"]
                display_content(content=content, request_id=request_id)
        st.session_state.messages.append({"role": "assistant", "content": content, "request_id": request_id})
    
    def display_content(content: List[Dict[str, str]], request_id: Optional[str] = None, message_index: Optional[int] = None) -> None:
        """Displays content for a chat message."""
        # Ensure messages key exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        message_index = message_index or len(st.session_state.messages)
        if request_id:
            with st.expander("Request ID", expanded=False):
                st.markdown(request_id)
        for item in content:
            if item["type"] == "text":
                st.markdown(item["text"])
            elif item["type"] == "suggestions":
                with st.expander("Suggestions", expanded=True):
                    for suggestion_index, suggestion in enumerate(item["suggestions"]):
                        if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                            st.session_state.active_suggestion = suggestion
            elif item["type"] == "sql":
                display_sql_query(item["statement"])
            else:
                st.write(f"Unsupported content type: {item['type']}")
    
    ###############################################################################
    # Query Execution and Caching for Data Explorer / Dashboard Pages
    ###############################################################################
    @st.cache_data(ttl=3600)
    def cached_run_sql_query(sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        conn = get_or_init_snowflake_conn()
        try:
            df = pd.read_sql(sql, conn)
            return df, None
        except Exception as exc:
            return None, str(exc)
    
    def display_sql_query(sql: str):
        """Displays the SQL query and its results along with chart options."""
        with st.expander("SQL Query", expanded=False):
            st.code(sql, language="sql")
        with st.expander("Results", expanded=True):
            with st.spinner("Running SQL..."):
                df, err = cached_run_sql_query(sql)
                if err:
                    st.error(f"SQL execution error: {err}")
                    return
                if df is None or df.empty:
                    st.write("Query returned no data.")
                    return
                tab_data, tab_chart = st.tabs(["Data 📄", "Charts 📈"])
                with tab_data:
                    st.dataframe(df)
                with tab_chart:
                    display_chart_tab(df, prefix="sql_chart")
    
    ###############################################################################
    # Chart Display Function (supports additional chart types)
    ###############################################################################
    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart"):
        """Allows user to select chart options and displays a chart with unique widget keys."""
        if len(df.columns) < 2:
            st.write("Not enough columns to chart.")
            return
    
        all_cols = list(df.columns)
        col1, col2, col3 = st.columns(3)
    
        # Use st.session_state.get(...) to obtain the previously selected values without reassigning them.
        default_x = st.session_state.get(f"{prefix}_x", all_cols[0])
        try:
            x_index = all_cols.index(default_x)
        except ValueError:
            x_index = 0
        x_col = col1.selectbox("X axis", all_cols, index=x_index, key=f"{prefix}_x")
    
        remaining_cols = [c for c in all_cols if c != x_col]
        default_y = st.session_state.get(f"{prefix}_y", remaining_cols[0])
        try:
            y_index = remaining_cols.index(default_y)
        except ValueError:
            y_index = 0
        y_col = col2.selectbox("Y axis", remaining_cols, index=y_index, key=f"{prefix}_y")
    
        chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
        default_type = st.session_state.get(f"{prefix}_type", "Line Chart")
        try:
            type_index = chart_options.index(default_type)
        except ValueError:
            type_index = 0
        chart_type = col3.selectbox("Chart Type", chart_options, index=type_index, key=f"{prefix}_type")
    
        # Display the chart based on the selected type.
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_line")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_bar")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_pie")
        elif chart_type == "Scatter Chart":
            fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_scatter")
        elif chart_type == "Histogram Chart":
            fig = px.histogram(df, x=x_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_hist")
    
    ###############################################################################
    # Dashboard Page
    ###############################################################################
    
    def dashboard_page():
        st.title("Dashboard")
        st.markdown("Configure your query and click **Apply** to see the results.")
        
        semantic_model = load_semantic_model()
        if not semantic_model:
            st.error("Semantic model could not be loaded.")
            return
    
        # --- Step 1: Select Metric (from fact table 'Opportunity_Line_Item') ---
        # Option 1: Hardcode the fact table based on your semantic model.
        fact_table_name = "Opportunity_Line_Item"  # Updated from "opportunity_facts"
        
        # Option 2 (alternative): Dynamically determine the fact table by checking for measures.
        # fact_tables = [t for t in semantic_model.get("tables", []) if t.get("measures")]
        # if fact_tables:
        #     fact_table_info = fact_tables[0]  # or allow user selection if multiple exist
        #     fact_table_name = fact_table_info["name"]
        # else:
        #     st.error("No fact table found in the semantic model.")
        #     return
    
        fact_table_info = get_table_info(semantic_model, fact_table_name)
        if not fact_table_info:
            st.error(f"Fact table '{fact_table_name}' not found in the semantic model.")
            return
        available_metrics = [m.get("name") for m in fact_table_info.get("measures", []) if m.get("name")]
        if not available_metrics:
            st.error("No metrics found in the fact table.")
            return
        selected_metric = st.sidebar.selectbox("Select Metric Name", available_metrics)
        metric_def = next((m for m in fact_table_info.get("measures", []) if m.get("name") == selected_metric), None)
        if not metric_def:
            st.error("Selected metric not found.")
            return
        metric_expr = metric_def.get("expr", selected_metric)
        metric_filters = ""
        if "filters" in metric_def:
            metric_filters = " AND ".join([f.strip() for f in metric_def["filters"] if f.strip()])
        
        # --- Step 2: Select Dimension Folder (from all tables except fact table) ---
        dimension_tables = [t.get("name") for t in semantic_model.get("tables", []) if t.get("name") != fact_table_name]
        if not dimension_tables:
            st.error("No dimension tables found in the semantic model.")
            return
        selected_dimension_folder = st.sidebar.selectbox("Select Dimension Folder", dimension_tables)
        dim_table_info = get_table_info(semantic_model, selected_dimension_folder)
        if not dim_table_info:
            st.error("Selected dimension folder not found.")
            return
    
        # --- Step 3: Select Attribute from the Dimension Folder ---
        available_attributes = [d.get("name") for d in dim_table_info.get("dimensions", []) if d.get("name")]
        if not available_attributes:
            st.error("No attributes found in the selected dimension folder.")
            return
        selected_attribute = st.sidebar.selectbox("Select Attribute", available_attributes)
        
        # --- Step 4: Optional Filters ---
        apply_date_filter = st.sidebar.checkbox("Apply Date Filter", value=False)
        if apply_date_filter:
            start_date = st.sidebar.date_input("Start Date")
            end_date = st.sidebar.date_input("End Date")
        else:
            start_date = end_date = None
    
        apply_year_filter = st.sidebar.checkbox("Apply Year Filter", value=False)
        if apply_year_filter:
            years = list(range(2025, 2014, -1))
            selected_year = st.sidebar.selectbox("Select Year", years)
        else:
            selected_year = None
    
        # --- Step 5: Construct the Natural Language Prompt ---
        prompt = f"Show me {selected_metric} by {selected_attribute}"
        if selected_year:
            prompt += f" for the year {selected_year}"
        if start_date and end_date:
            prompt += f" from {start_date} to {end_date}"
        
        # --- Step 6: Run Query and Store Results ---
        if st.sidebar.button("Apply"):
            progress = st.empty()
            messages = [
                "🧠 Analyzing your request...",
                "🔍 Querying the semantic model...",
                "⚙️ Generating optimized SQL...",
                "❄️ Executing in Snowflake...",
                "📊 Preparing visualizations..."
            ]
            
            start_time = time.time()
            frame_delay = 0.5  # Faster spinning animation
            animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            message_duration = 3  # Cycle through messages every 3 seconds
    
            # Capture the connection’s token on the main thread.
            conn = get_or_init_snowflake_conn()
            token = conn.rest.token
    
            # Start processing in a separate thread.
            response = None
            processing_complete = False
            
            def get_response(token):
                nonlocal response, processing_complete
                try:
                    response = send_message(prompt, token=token)
                except Exception as e:
                    response = {"error": str(e)}
                processing_complete = True
                
            thread = threading.Thread(target=get_response, args=(token,))
            thread.start()
            
            # Animation loop while waiting for the API call to complete.
            while not processing_complete:
                elapsed = time.time() - start_time
                current_msg_index = int(elapsed // message_duration) % len(messages)
                frame = animation_frames[int((elapsed % 0.5) * 10) % len(animation_frames)]
                progress.markdown(
                    f"<div style='display: flex; align-items: center; gap: 0.5rem;'>"
                    f"<span style='color: #4f8bf9; font-size: 24px;'>{frame}</span>"
                    f"<span style='font-size: 18px;'>{messages[current_msg_index]}</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                time.sleep(frame_delay)
            
            progress.empty()
            
            # Store the SQL result (if any) without displaying Response or Generated SQL.
            if response is None or "error" in response:
                error_message = response.get("error", "Unknown error occurred.") if response else "No response received."
                st.error(f"Error: {error_message}")
                return
                
            sql_statement = next((item.get("statement") for item in response.get("message", {}).get("content", []) if item.get("type") == "sql"), None)
            if sql_statement:
                df, err = cached_run_sql_query(sql_statement)
                if err:
                    st.error(f"SQL execution error: {err}")
                elif df is None or df.empty:
                    st.write("No data returned for the generated SQL.")
                else:
                    st.session_state.dashboard_df = df
                    st.session_state.dashboard_sql = sql_statement
            else:
                st.info("No SQL statement was returned by the chat service.")
        
        # If results exist, display only the Data Preview and Charts.
        if "dashboard_df" in st.session_state and st.session_state.dashboard_df is not None:
            st.markdown("### Data Preview")
            st.dataframe(st.session_state.dashboard_df)
            st.markdown("### Charts")
            display_chart_tab(st.session_state.dashboard_df, prefix="dashboard_chart")
    
    
    ###############################################################################
    # Data Explorer Page
    ###############################################################################
    def data_explorer_page():
        st.title("Data Explorer")
        st.markdown("Enter a SQL query to run against Snowflake:")
        with st.form("sql_form"):
            sql_query = st.text_area("SQL Query", height=150)
            submitted = st.form_submit_button("Run Query")
            if submitted and sql_query.strip():
                display_sql_query(sql_query)
        st.markdown("Use the data editor above to interactively edit results.")
    
    ###############################################################################
    # Chat Page
    ###############################################################################
    def chat_page():
        st.title("Cortex Analyst - Chat")
        st.markdown(f"Semantic Model: `{FILE}`")
        # Initialize session state variables if they do not exist.
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "suggestions" not in st.session_state:
            st.session_state.suggestions = []
        if "active_suggestion" not in st.session_state:
            st.session_state.active_suggestion = None
        for message_index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                display_content(content=message["content"], request_id=message.get("request_id"), message_index=message_index)
        if user_input := st.chat_input("Message Snowflake"):
            process_message(prompt=user_input)
        if st.session_state.active_suggestion:
            process_message(prompt=st.session_state.active_suggestion)
            st.session_state.active_suggestion = None
    
    ###############################################################################
    # Settings Page
    ###############################################################################
    def settings_page():
        st.title("Settings")
        st.markdown("Adjust your application settings here.")
        new_model = st.text_input("Semantic Model Path", value=AVAILABLE_SEMANTIC_MODELS_PATHS[0])
        if st.button("Update Model"):
            st.session_state.selected_semantic_model_path = new_model
            st.success("Semantic model updated!")
    
    ###############################################################################
    # Main Application with Query Parameter Navigation and Fixed Bottom Nav
    ###############################################################################
    def main():
        # Define valid pages.
        valid_pages = {"Chat", "Data Explorer", "Dashboard", "Settings"}
        # Use st.query_params (property) to get current query parameters.
        params = st.query_params
        page_list = params.get("page", ["Chat"])
        if isinstance(page_list, list):
            current_page = page_list[0]
        else:
            current_page = page_list
        # Validate the page.
        if current_page not in valid_pages:
            current_page = "Chat"
    
        # Add extra bottom padding so content is not hidden behind the fixed nav bar.
        st.markdown("""
        <style>
        .reportview-container .main .block-container {
            padding-bottom: 140px;
        }
        </style>
        """, unsafe_allow_html=True)
    
        # Render the current page.
        if current_page == "Chat":
            chat_page()
        elif current_page == "Data Explorer":
            data_explorer_page()
        elif current_page == "Dashboard":
            dashboard_page()
        elif current_page == "Settings":
            settings_page()
        else:
            st.error("Invalid page selection!")
    
        # Include Font Awesome for icons.
        fa_link = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">'
        st.markdown(fa_link, unsafe_allow_html=True)
    
        # Define the chat input color; change this value to match your chat input.
        chat_input_color = "#262730"  # Default chat input background is white.
    
        # Create a helper function to determine the selected nav item.
        selected = lambda page: "selected" if current_page == page else ""
        
        # Build the navigation HTML with a light background using the chat input color,
        # icons added for each menu option, and increased padding for the selected option.
        nav_html = f"""
        <style>
        .fixed-bottom-nav {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: {chat_input_color};
            padding: 10px 0;
            text-align: center;
            z-index: 1000;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        }}
        .fixed-bottom-nav a {{
            color: #FFFFFF;
            margin: 0 20px;
            font-size: 18px;
            text-decoration: none;
            padding: 16px 24px;
            border-radius: 4px;
        }}
        .fixed-bottom-nav a:hover {{
            background-color: #e0e0e0;
        }}
        .fixed-bottom-nav a.selected {{
            background-color: #2196F3;
            color: white;
            padding: 16px 32px;  /* Increased horizontal padding for selected option */
        }}
        </style>
        <div class="fixed-bottom-nav">
        <a href="?page=Chat" target="_self" class="{selected('Chat')}">
            <i class="fa fa-comments" aria-hidden="true"></i> Chat
        </a>
        <a href="?page=Data%20Explorer" target="_self" class="{selected('Data Explorer')}">
            <i class="fa fa-database" aria-hidden="true"></i> Data Explorer
        </a>
        <a href="?page=Dashboard" target="_self" class="{selected('Dashboard')}">
            <i class="fa fa-tachometer" aria-hidden="true"></i> Dashboard
        </a>
        <a href="?page=Settings" target="_self" class="{selected('Settings')}">
            <i class="fa fa-cog" aria-hidden="true"></i> Settings
        </a>
        </div>
        """
        st.markdown(nav_html, unsafe_allow_html=True)
    
    if __name__ == "__main__":
        main()