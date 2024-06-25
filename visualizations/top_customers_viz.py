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
#Visualization of Customer_details
def customer_analysis_app(df):
    """Creates a Streamlit app with tabs for analyzing customer data using plots."""

    st.title("Customer Sales Analysis")

    tab1, tab2, tab3 = st.tabs(["Top Customers", "Territory Analysis", "Payment Terms Analysis"])

    with tab1:
        st.subheader("Top 10 Customers")
        top_10_customers = df.groupby('Name')['Total sales'].sum().nlargest(10).reset_index()

        fig = go.Figure(data=go.Bar(
            x=top_10_customers['Name'],
            y=top_10_customers['Total sales'],
            marker_color=px.colors.qualitative.Light24,
            text=top_10_customers['Total sales'].apply(lambda x: f'${x:,.2f}'),
            textposition='outside',
            hovertemplate='<b>Customer:</b> %{x}<br><b>Total Sales:</b> $%{y:,.2f}<extra></extra>'
        ))
        fig.update_layout(
            title="Top 10 Customers by Total Sales",
            xaxis_title="Customer",
            yaxis_title="Total Sales",
            template="plotly_white",
            height=550  # Set the height of the plot
    )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Sales by Territory")
        territory_sales = df.groupby('Territory')['Total sales'].sum().reset_index()

        fig = go.Figure(data=go.Pie(
            labels=territory_sales['Territory'],
            values=territory_sales['Total sales'],
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Pastel),
            hovertemplate='<b>Territory:</b> %{label}<br><b>Total Sales:</b> $%{value:,.2f}<br><b>Percentage:</b> %{percent}<extra></extra>',
            textinfo='percent+label'
        ))
        fig.update_layout(
            title="Sales Distribution by Territory",
        height=550  # Set the height of the plot
    )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Sales by Payment Terms")
        payment_terms_sales = df.groupby('Payment terms')['Total sales'].sum().reset_index()

        fig = go.Figure(data=go.Bar(
            x=payment_terms_sales['Payment terms'],
            y=payment_terms_sales['Total sales'],
            marker_color=px.colors.qualitative.Pastel,
            text=payment_terms_sales['Total sales'].apply(lambda x: f'${x:,.2f}'),
            textposition='outside',
            hovertemplate='<b>Payment Terms:</b> %{x}<br><b>Total Sales:</b> $%{y:,.2f}<extra></extra>'
        ))
        fig.update_layout(
            title="Sales Distribution by Payment Terms",
            xaxis_title="Payment Terms",
            yaxis_title="Total Sales",
            template="plotly_white",
            height=550  # Set the height of the plot
    )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Understanding Customer Behavior: Sales Insights

    This dashboard provides an overview of customer sales patterns, focusing on your top-performing customers, sales distribution across different territories, and a breakdown of sales by payment terms. Use this information to identify key customer segments, optimize sales strategies, and improve cash flow management.
    """)
#--------------------------bar_plot_with_percentages- SUB FUNCTION-------------------------------------
def create_bar_plot_with_percentages(df, col="Payment terms"):
    counts = df[col].value_counts().sort_values(ascending=False)
    percentages = (counts / len(df)) * 100
    df_plot = pd.DataFrame({'Category': counts.index, 'Count': counts.values, 'Percentage': percentages})

    fig = go.Figure(data=go.Bar(
        x=df_plot['Category'],
        y=df_plot['Count'],
        marker_color=px.colors.qualitative.Pastel,
        text=df_plot['Percentage'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>Category:</b> %{x}<br><b>Count:</b> %{y}<br><b>Percentage:</b> %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Distribution by {col.title()}",
        xaxis_title=col.title(),
        yaxis_title="Count",
        template="plotly_white",
        title_x=0.5,
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        height=550  # Set the height of the plot
    )
        
    
    return fig

def interactive_bar_plot_app(df):
    st.title("Distribution Analysis") # More concise title

    column_options = df.select_dtypes(include='object').columns
    selected_column = st.selectbox("Select a Category", column_options)

    fig = create_bar_plot_with_percentages(df, selected_column)
    st.plotly_chart(fig)

    st.markdown("""
    ## Understanding Distribution Patterns

    This interactive bar chart allows you to analyze the distribution of data across different categories within your dataset. Explore various categorical columns to uncover patterns, identify dominant categories, and gain insights into the composition of your data. 
    """)
#Data distribution visualization
def create_non_zero_sales_grouped_plot(df, sales_col='Total sales', threshold=500):
    df_filtered = df[df[sales_col] > 0]
    df_below_threshold = df_filtered[df_filtered[sales_col] <= threshold]
    df_above_threshold = df_filtered[df_filtered[sales_col] > threshold]
    counts_below = df_below_threshold[sales_col].value_counts().sort_index()
    count_above = df_above_threshold[sales_col].count()
    values = counts_below.index.tolist() + [f"{threshold}+"]
    counts = counts_below.values.tolist() + [count_above]
    df_plot = pd.DataFrame({'Sales Value': values, 'Count': counts})

    fig = go.Figure(data=go.Scatter(
        x=df_plot['Sales Value'],
        y=df_plot['Count'],
        mode='lines+markers',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        text=df_plot['Count'],
        hovertemplate='<b>Sales Value:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Distribution of Non-Zero Total Sales",
        title_x=0, 
        xaxis_title="Value of Total Sales", 
        yaxis_title="Number of Entries",
        template="plotly_white"
    )
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Sales Distribution: Identifying Patterns and Outliers

    This line chart illustrates the distribution of non-zero total sales values, providing a visual representation of sales frequencies. Analyze the shape of the line to identify common sales value ranges, potential outliers (sudden spikes or drops), and gain a better understanding of the overall sales distribution.
    """)

#Distribution of customer groups by city
def create_bar_plot_with_legend(df, city_col, group_col, title):
    grouped_data = df.groupby([city_col, group_col]).size().unstack().fillna(0)
    data = [
        go.Bar(
            x=grouped_data.index,
            y=grouped_data[group],
            name=group,
            hovertemplate='<b>City:</b> %{x}<br><b>Group:</b> ' + group + '<br><b>Count:</b> %{y}<extra></extra>'
        ) for group in grouped_data.columns
    ]

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        xaxis_title="City",
        yaxis_title="Number of Clients",
        barmode='group',
        template="plotly_white",
        legend_title_text="Group",
        title_x=0
    )
    return fig

def interactive_group_distribution_app(df, group_col='Group', city_col='Billing city'):
    st.title("Customer Group Distribution")  # Concise title

    most_frequent_city = df[city_col].value_counts().index[0]

    data_all_cities = df.copy()
    data_without_frequent_city = df[df[city_col] != most_frequent_city]

    tab1, tab2 = st.tabs(["All Cities", f"Excluding {most_frequent_city}"])

    with tab1:
        fig1 = create_bar_plot_with_legend(data_all_cities, city_col, group_col, "Client Group Distribution by City")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = create_bar_plot_with_legend(data_without_frequent_city, city_col, group_col, f"Client Group Distribution (Excluding {most_frequent_city})")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ## Geographic Insights: Customer Group Distribution

    This interactive visualization explores the distribution of customer groups across different cities. Analyze how customer groups are concentrated or spread out geographically, identify key markets, and uncover potential opportunities for expansion or targeted marketing efforts. 
    """)