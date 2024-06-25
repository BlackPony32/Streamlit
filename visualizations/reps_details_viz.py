import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

def preprocess_data(data):
    """
    Data preprocessing: data type conversion and cleaning.

    Args:
        data: A Pandas DataFrame with the source data.

    Returns:
        Pandas DataFrame with the processed data.
    """

    # Identify numeric columns automatically
    numeric_cols = data.select_dtypes(include=np.number).columns

    # Process numeric columns
    for col in numeric_cols:
        # Check for missing values (NaN)
        if np.isnan(data[col]).any():
            # Fill missing values with 0 (you can choose another strategy)
            data[col].fillna(0, inplace=True)
            print(f"Warning: Column '{col}' contains missing values (NaN). Filled with 0.")

    # Remove currency symbols and thousands separators
    data[numeric_cols] = data[numeric_cols].replace('[$,]', '', regex=True).astype(float)

    return data

def analyze_sales_rep_efficiency(df_pd):
    """Analyzes sales representative efficiency and displays a pie chart."""

    df_pd = df_pd.copy()

    def convert_hours_to_numeric(time_str):
        try:
            hours, minutes = map(int, time_str.split('h '))
            return hours + minutes / 60
        except ValueError:
            return pd.NA

    df_pd['Total working hours'] = df_pd['Total working hours'].apply(convert_hours_to_numeric)
    df_pd["Visits per Working Hour"] = df_pd["Total visits"] / df_pd["Total working hours"]
    df_pd["Customers per Visit"] = df_pd["Assigned customers"] / df_pd["Total visits"]

    grouped = df_pd.groupby("Role")[["Total visits"]].sum().reset_index()

    fig = go.Figure(data=[go.Pie(
        labels=grouped['Role'],
        values=grouped['Total visits'],
        hole=0.3,
        textinfo='percent+label',
        marker=dict(colors=px.colors.qualitative.Set2),
        hovertemplate="<b>Role:</b> %{label}<br>" +
                      "<b>Total Visits:</b> %{value}<br>" +
                      "<b>Percentage:</b> %{percent:.1%}<extra></extra>"
    )])

    fig.update_layout(
        title="Distribution of Total Visits by Role",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_font=dict(size=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Visit Distribution: Understanding Role Contributions

    This interactive donut chart provides a clear picture of how total visits are distributed among your sales roles. By visualizing these proportions, you can gain a better understanding of each role's contribution to overall sales efforts. 
    """)


#Visualizing Customer Engagement: Active Customers vs. Total Visits
def plot_active_customers_vs_visits(df_pd):
    sales_data = df_pd[df_pd["Role"] == "SALES"]

    fig = go.Figure()

    for name in sales_data["Name"].unique():
        rep_data = sales_data[sales_data["Name"] == name]
        fig.add_trace(go.Scatter(
            x=rep_data["Active customers"],
            y=rep_data["Total visits"],
            mode='markers',
            marker=dict(size=10),
            name=name,
            hovertemplate = "<b>Active Customers:</b> %{x}<br>" + "<b>Total Visits:</b> %{y}<extra></extra>"
        ))

    fig.update_layout(
        title="Active Customers vs. Total Visits (Sales Reps)",
        xaxis_title="Active Customers",
        yaxis_title="Total Visits",
        colorway=px.colors.qualitative.Set2,  # Set color palette
        legend_title_text="Name",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ## Customer Engagement vs. Sales Activity

    This scatter plot explores the relationship between the number of active customers a sales representative handles and their total number of visits.  By analyzing individual performance and overall trends, you can identify opportunities to optimize sales strategies, resource allocation, and potentially set more effective goals. 
    """)


#Travel Distance vs. Number of Visits
def plot_travel_efficiency_line(df_pd):
    """Plots a scatter plot to visualize travel efficiency."""

    # Copy the dataframe to avoid modifying the original data
    df_pd = df_pd.copy()

    # Extract numeric part from "Total travel distance"
    df_pd["Total travel distance"] = df_pd["Total travel distance"].str.extract(r'(\d+\.?\d*)').astype(float)

    # Create the scatter plot
    fig = go.Figure()

    for role in df_pd["Role"].unique():
        role_data = df_pd[df_pd["Role"] == role]
        fig.add_trace(go.Scatter(
            x=role_data["Total travel distance"],
            y=role_data["Total visits"],
            mode='markers',
            name=role,
            hovertemplate="<b>Name:</b> %{text}<br>" +
                          "<b>Total Travel Distance:</b> %{x}<br>" +
                          "<b>Total Visits:</b> %{y}<extra></extra>",
            text=role_data["Name"]
        ))

    fig.update_layout(
        title="Travel Efficiency: Distance vs. Visits",
        xaxis_title="Total Travel Distance (miles)",
        yaxis_title="Total Visits",
        title_font_size=20,
        legend_title_text="Role",
        colorway=px.colors.qualitative.Set1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, b=40, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Travel Efficiency: Distance vs. Visits

    This scatter plot shows the relationship between total travel distance and the number of visits. Each point represents a team member, colored by their role.

    Use this plot to identify trends, outliers, and opportunities to optimize travel routes.
    """)


#Pure work time per Employee
def analyze_work_hours_and_distance(df_pd):
    """
    Calculates clear work hours and visualizes both clear work hours and
    total travel distance in separate tabs.
    """
    df_pd = df_pd.copy()

    def parse_time(time_str):
        if pd.isna(time_str):
            return 0
        import re
        match = re.match(r'(\d+)h\s*(\d+)m', time_str)
        if match:
            h, m = map(int, match.groups())
        else:
            match = re.match(r'(\d+)m', time_str)
            if match:
                m = int(match.group(1))
                h = 0
            else:
                h, m = 0, 0
        return h + m / 60

    df_pd['Total working hours'] = df_pd['Total working hours'].apply(parse_time)
    df_pd['Total break hours'] = df_pd['Total break hours'].apply(parse_time)
    df_pd['Pure Work Hours'] = df_pd['Total working hours'] - df_pd['Total break hours']
    df_pd = df_pd.sort_values(by='Pure Work Hours', ascending=False).head(10)

    # Extract numeric part from "Total travel distance" 
    df_pd["Total travel distance"] = df_pd["Total travel distance"].str.extract('(\d+\.?\d*)').astype(float)

    tab1, tab2 = st.tabs(["Pure Work Hours", "Total Travel Distance"])

    with tab1:
        st.subheader("Top 10 Employees by Work Hours")
        fig = go.Figure(go.Bar(
            x=df_pd['Name'],
            y=df_pd['Pure Work Hours'],
            text=[f"{x:.1f}h" for x in df_pd['Pure Work Hours']],
            textposition='outside',
            marker_color=px.colors.qualitative.Light24,
            name="Pure Work Hours"
        ))
        fig.update_layout(
            title="Top 10 Employees by Work Hours",
            xaxis_title="Employee",
            yaxis_title="Hours",
            xaxis_tickangle=45,
            height=500,
            legend_title="Metrics",
            margin=dict(l=40, r=40, b=40, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Top 10 Employees by Travel Distance")
        fig = go.Figure(go.Bar(
            x=df_pd['Name'],
            y=df_pd['Total travel distance'],
            text=[f"{x:.1f} mi" for x in df_pd['Total travel distance']],
            textposition='outside',
            marker_color=px.colors.qualitative.Light24,
            name="Total Travel Distance"
        ))
        fig.update_layout(
            title="Top 10 Employees by Travel Distance",
            xaxis_title="Employee",
            yaxis_title="Miles",
            xaxis_tickangle=45,
            height=500,
            legend_title="Metrics",
            margin=dict(l=40, r=40, b=40, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Workload and Travel: Insights into Top Performers

    These bar charts, separated into tabs for easy navigation, highlight the top 10 employees based on pure work hours and total travel distance. Use these visualizations to identify potential workload imbalances, analyze travel patterns, and explore ways to optimize efficiency and resource allocation.
    """)


#Total Visits vs. Total Photos Taken
def plot_visits_vs_photos_separate(df_pd):
    """Plots separate scatter plots for visits vs. photos for each role."""

    # Get unique roles
    roles = df_pd['Role'].unique()

    # Create tabs for each role
    tabs = st.tabs([role for role in roles]) 

    for i, role in enumerate(roles):
        with tabs[i]:
            st.subheader(f"Visits vs. Photos ({role})")
            role_data = df_pd[df_pd['Role'] == role]
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=role_data["Total visits"],
                y=role_data["Total photos"],
                mode='markers',
                marker=dict(color=px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)]),
                text=role_data["Name"],
                hovertemplate="<b>%{text}</b><br>Total Visits: %{x}<br>Total Photos: %{y}<extra></extra>"
            ))

            fig.update_layout(
                title=f"Visits vs. Photos ({role})",
                xaxis_title="Total Visits",
                yaxis_title="Total Photos",
                template="plotly_white",
                height=500,
                legend_title="Role",
                margin=dict(l=40, r=40, b=40, t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

    # Provide a brief markdown explanation
    st.markdown("""
    ## Visits vs. Photos: Exploring the Relationship

    These scatter plots analyze the relationship between total visits and the number of photos taken by team members for each role. This visualization helps to understand engagement levels and photo-taking patterns.
    """)


#Exploring Customer Distribution Across Sales Representatives
def analyze_customer_distribution(df_pd):
    # Filter sales data
    sales_data = df_pd[df_pd["Role"] == "SALES"].copy()

    # Group by sales representative and sum the number of assigned customers
    customer_distribution = sales_data.groupby("Name")["Assigned customers"].sum().reset_index()

    # Create a bar plot to visualize the distribution of assigned customers
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=customer_distribution["Name"],
        y=customer_distribution["Assigned customers"],
        marker=dict(color=customer_distribution["Assigned customers"], colorscale='Viridis'),
        text=customer_distribution["Assigned customers"],
        hovertemplate="<b>%{x}</b><br>Assigned Customers: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="Assigned Customers per Sales Representative",
        xaxis_title="Sales Representative",
        yaxis_title="Number of Assigned Customers",
        title_font_size=20,
        xaxis_tickangle=-45,
        template="plotly_white",
        coloraxis_colorbar=dict(title="Assigned Customers")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed markdown explanation
    st.markdown("""
    ## Understanding Customer Allocation

    The bar plot above shows the number of assigned customers per sales representative. This visualization helps to identify potential imbalances in workload or variations in customer assignments. By analyzing this distribution, you can make informed decisions to optimize sales territories and ensure a more balanced workload across your sales team.
    """)
