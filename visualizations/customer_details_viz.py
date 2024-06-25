import plotly.express as px
import pandas as pd
import streamlit as st
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
#Visualization Customer_details
def plot_orders_and_sales_plotly(df, group_col='Group'):
    orders = df.groupby(group_col)["Total orders"].sum()
    sales = df.groupby(group_col)["Total sales"].sum()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=orders.index,
        y=orders,
        name='Total Orders',
        marker_color='skyblue',
        marker_line_color='steelblue',
        marker_line_width=1,
        texttemplate='%{y:.2s}',
        textposition='outside',
        hovertemplate='<b>Customer:</b> %{x}<br><b>Total Orders:</b> %{y}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=sales.index,
        y=sales,
        name='Total Sales',
        marker_color='coral',
        marker_line_color='darkred',
        marker_line_width=1,
        texttemplate='%{y:.2s}',
        textposition='outside',
        hovertemplate='<b>Customer:</b> %{x}<br><b>Sales Amount:</b> $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        barmode='group',
        title='Comparison of Total Orders and Sales by Customer Group',
        xaxis=dict(
            title='Customer Group',
            tickangle=45,
            title_font_size=12
        ),
        yaxis=dict(
            title='Count/Amount',
            title_font_size=12
        ),
        legend_title_text='Metrics',
        showlegend=True,
        height=550  # Set the height of the plot
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
## Customer Group Performance: Orders & Sales

These visualizations compare total orders and sales for each customer group, revealing valuable insights about your most important customer segments. Use this data to identify high-value customers, understand purchasing behavior, and optimize your sales and marketing strategies.
""")

#________________________________________________________________
def bar_plot_sorted_with_percentages(df, col='Payment terms'):
    counts = df[col].value_counts().sort_values(ascending=False)
    percentages = (counts / len(df)) * 100
    df_plot = pd.DataFrame({'Category': counts.index, 'Count': counts.values, 'Percentage': percentages})

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_plot['Category'],
        y=df_plot['Count'],
        text=df_plot['Percentage'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>Category:</b> %{x}<br><b>Count:</b> %{y}<br><b>Percentage:</b> %{text}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Distribution of Clients by {col}",
        title_x=0,
        xaxis=dict(
            title=col,
            tickangle=45,
            tickfont_size=10
        ),
        yaxis=dict(
            title='Count'
        ),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        height=550  # Set the height of the plot
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
## Payment Terms: Understanding Client Preferences

This visualization shows how your clients are distributed across different payment terms. Use these insights to optimize your cash flow management, tailor your offerings to different customer segments, and make informed decisions about credit risk.
""")
#Data distribution visualization function
def create_interactive_non_zero_sales_plot(df, sales_col='Total sales', threshold=500):
    df_filtered = df[df[sales_col] > 0]
    df_below_threshold = df_filtered[df_filtered[sales_col] <= threshold]
    df_above_threshold = df_filtered[df_filtered[sales_col] > threshold]
    counts_below = df_below_threshold[sales_col].value_counts().sort_index()
    count_above = df_above_threshold[sales_col].count()
    values = counts_below.index.tolist() + [f"{threshold}+"]
    counts = counts_below.values.tolist() + [count_above]
    df_plot = pd.DataFrame({'Sales Value': values, 'Count': counts})

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot['Sales Value'],
        y=df_plot['Count'],
        mode='lines+markers',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hovertemplate='<b>Sales Value:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Distribution of Non-Zero {sales_col}",
        title_x=0,
        xaxis=dict(
            title="Value of Total Sales"
        ),
        yaxis=dict(
            title="Number of Entries"
        )
    )

    st.plotly_chart(fig)
    st.markdown("""
## Sales Value Distribution: Insights for Growth

This visualization reveals how non-zero total sales values are distributed. Use these insights to refine your pricing and promotion strategies, enhance sales forecasting, optimize product development, and effectively manage potential risks.
""")
#Average total sales by customer group and billing state
def create_interactive_average_sales_heatmap(df):
    df['Total sales'] = df['Total sales'].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    average_sales = df.groupby(["Group", "Billing state"])["Total sales"].mean().unstack()

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=average_sales.values,
        x=average_sales.columns,
        y=average_sales.index,
        colorscale='YlGnBu',
        colorbar=dict(title="Average Sales"),
        hovertemplate='<b>Customer Group:</b> %{y}<br><b>Billing State:</b> %{x}<br><b>Average Sales:</b> $%{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title="Average Total Sales by Customer Group and State",
        xaxis=dict(
            title="Billing State"
        ),
        yaxis=dict(
            title="Customer Group"
        ),
        height=550  # Set the height of the plot
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
## Sales Performance: Customer Groups & Geographic Breakdown

This heatmap reveals average total sales across customer groups and states. Use it to:

- **Pinpoint high-performing segments and regions.**
- **Identify areas for growth and optimization.**
- **Develop targeted sales and marketing strategies.**
- **Efficiently allocate resources and manage inventory.**
""")