import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np
from plotly.colors import sequential
from plotly.colors import qualitative
import plotly.graph_objects as go
import plotly.colors as colors
#Data preprocess
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

#Distribution of unordered products across different categories
def create_unordered_products_by_category_plot(df):
    category_counts = df['Category name'].value_counts()

    fig = go.Figure(go.Bar(
        x=category_counts.index,
        y=category_counts.values,
        marker_color=colors.qualitative.Prism,
        hovertemplate="<b>Category:</b> %{x}<br><b>Unordered Products:</b> %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Unordered Products by Category",
        xaxis_title="Category",
        yaxis_title="Number of Unordered Products",
        xaxis_tickangle=45,
        template="plotly_white",
        hovermode="closest"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Unordered Products: A Category-Based View

    This bar chart displays the number of unordered products in each category. Use this visualization to identify potential stock shortages, prioritize reordering, and refine your inventory management strategies.
    """)

#Visualizes the distribution of available cases for unordered products using a histogram
def create_available_cases_distribution_plot(df):
    stock_levels = {
        "Low Stock (0-10)": (0, 10),
        "Medium Stock (11-50)": (11, 50),
        "High Stock (51+)": (51, float('inf'))
    }

    def assign_stock_level(stock):
        for level, (min_val, max_val) in stock_levels.items():
            if min_val <= stock <= max_val:
                return level

    df["Stock Level"] = df['Available cases (QTY)'].apply(assign_stock_level)
    stock_level_counts = df["Stock Level"].value_counts()

    fig = go.Figure(go.Pie(
        values=stock_level_counts.values,
        labels=stock_level_counts.index,
        hole=0.3,
        marker=dict(colors=colors.qualitative.Pastel),
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>Stock Level:</b> %{label}<br><b>Number of Products:</b> %{value}<br><b>Percentage:</b> %{percent}<extra></extra>"
    ))

    fig.update_layout(
        title="Distribution of Products by Stock Level",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Inventory Snapshot: Stock Level Distribution

    This donut chart presents a clear picture of how your products are distributed across different stock levels. It provides a quick assessment of potential stock shortages ("Low Stock"), healthy inventory levels ("Medium Stock"), and potential overstocking ("High Stock"). 
    """)

def price_vs_available_cases_app(df):
    st.title("Average Available Cases by Price Range and Category")

    category_options = df['Category name'].unique()
    selected_category = st.selectbox("Select a Category", category_options)

    df['Price Range'] = pd.cut(df['Retail price'], bins=3, labels=["Low", "Medium", "High"])
    average_cases_data = df[df['Category name'] == selected_category].groupby(['Price Range'])['Available cases (QTY)'].mean()

    fig = go.Figure(go.Bar(
        x=average_cases_data.index,
        y=average_cases_data.values,
        marker_color=colors.qualitative.Pastel,
        hovertemplate="<b>Price Range:</b> %{x}<br><b>Average Available Cases:</b> %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Average Available Cases for {selected_category}",
        xaxis_title="Retail Price Range",
        yaxis_title="Average Available Cases",
        template="plotly_white",
        hovermode="closest"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Inventory by Price: Analyzing Availability 

    This bar chart displays the average available cases for the selected product category across different retail price ranges. Use this information to identify potential stock imbalances within price ranges and make informed decisions about inventory management and pricing strategies. 
    """)

#Visualizes the relationship between wholesale and retail prices for unordered products
def create_wholesale_vs_retail_price_scatter(df):
    tab1, tab2 = st.tabs(["Available Cases vs Profit Margin", "Wholesale vs Retail Price"])

    df["Profit Margin %"] = (df['Retail price'] - df['Wholesale price']) / df['Wholesale price'] * 100

    # Create a color map for categories
    unique_categories = df['Category name'].unique()
    color_map = dict(zip(unique_categories, colors.qualitative.Plotly[:len(unique_categories)]))

    with tab1:
        traces = []
        for category in unique_categories:
            df_category = df[df['Category name'] == category]
            traces.append(go.Scatter(
                x=df_category['Available cases (QTY)'],
                y=df_category["Profit Margin %"],
                mode='markers',
                name=category,
                marker=dict(color=color_map[category], size=8),
                text=df_category['Category name'],
                hovertemplate="<b>Category:</b> %{text}<br><b>Available Cases:</b> %{x}<br><b>Profit Margin:</b> %{y:.2f}%<br><b>Retail Price:</b> $%{customdata[0]:.2f}<br><b>Wholesale Price:</b> $%{customdata[1]:.2f}<extra></extra>",
                customdata=df_category[['Retail price', 'Wholesale price']]
            ))

        fig1 = go.Figure(data=traces)
        
        fig1.update_layout(
            title="Available Cases vs. Profit Margin",
            xaxis_title="Available Cases",
            yaxis_title="Profit Margin (%)",
            template="plotly_white",
            hovermode="closest",
            font=dict(size=12),
            title_font_size=14,
            legend_title_text='Category',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)"
            )
        )
        
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure(go.Scatter(
            x=df['Wholesale price'],
            y=df['Retail price'],
            mode='markers',
            marker=dict(
                color=[color_map[cat] for cat in df['Category name']],
                size=8
            ),
            text=df['Category name'],
            hovertemplate="<b>Category:</b> %{text}<br><b>Wholesale Price:</b> $%{x:.2f}<br><b>Retail Price:</b> $%{y:.2f}<extra></extra>"
        ))
        
        fig2.update_layout(
            title="Wholesale vs. Retail Price",
            xaxis_title="Wholesale Price",
            yaxis_title="Retail Price",
            template="plotly_white",
            hovermode="closest",
            legend_title_text='Category',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.5)"
            )
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ## Profit & Pricing: Analyzing Relationships

    These scatter plots help you explore the connections between available cases, profit margins, wholesale prices, and retail prices. Use these visualizations to identify potential trends, outliers, and opportunities for optimizing pricing and inventory strategies. 
    """)

def df_unordered_products_per_category_and_price_range(df, category_col='Category name', retail_price_col='Retail price'):
    price_ranges = [0, 20, 40, 60, 80, 100, float('inf')]  # Added 100+ range
    price_labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100+"]
    df['Price Range'] = pd.cut(df[retail_price_col], bins=price_ranges, labels=price_labels)
    result = df.groupby([category_col, 'Price Range']).size().unstack(fill_value=0)
    
    fig = go.Figure(go.Heatmap(
        z=result.values,
        x=result.columns,
        y=result.index,
        colorscale='Reds',
        hovertemplate="<b>Category:</b> %{y}<br><b>Price Range:</b> %{x}<br><b>Number of Products:</b> %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Unordered Products: Category and Price View",
        xaxis_title="Price Range",
        yaxis_title="Category",
        xaxis_side="top",
        yaxis_autorange='reversed'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Unordered Products: Insights by Category and Price

    This heatmap reveals the distribution of unordered products across different categories and price ranges. Deeper colors represent a higher concentration of unordered items. Analyze this visualization to identify potential stock shortages, prioritize reordering based on price and category, and optimize inventory management strategies. 
    """)