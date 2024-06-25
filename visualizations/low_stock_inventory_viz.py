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

def low_stock_analysis_app(df):
    st.title("Low Stock Inventory Analysis")

    tab1, tab2 = st.tabs(["Distribution by Category", "Price vs. Quantity"])

    with tab1:
        category_counts = df.groupby("Category name")["Product name"].count().reset_index()
        fig1 = go.Figure(go.Pie(
            labels=category_counts["Category name"],
            values=category_counts["Product name"],
            hole=0.3,
            textinfo='percent+label',
            marker=dict(colors=px.colors.qualitative.Pastel),
            hovertemplate="<b>Category:</b> %{label}<br><b>Count:</b> %{value}<br><b>Percentage:</b> %{percent}<extra></extra>"
        ))
        fig1.update_layout(
            title="Low Stock Items by Category",
            legend=dict(title="Category", orientation="h", y=1.1, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        df_positive_qty = df[df['Available cases (QTY)'] > 0]  
        fig2 = go.Figure(data=go.Scatter(
            x=df_positive_qty["Wholesale price"],
            y=df_positive_qty["Available cases (QTY)"],
            mode='markers',
            marker=dict(
                size=df_positive_qty["Available cases (QTY)"],
                color=df_positive_qty["Wholesale price"],
                #colorscale='Pastel',
                showscale=True
            ),
            text=df_positive_qty['Product name'],
            hovertemplate="<b>%{text}</b><br>Wholesale Price: %{x}<br>Available Cases (QTY): %{y}<extra></extra>"
        ))
        fig2.update_layout(
            title="Wholesale Price vs. Available Quantity",
            xaxis_title="Wholesale Price",
            yaxis_title="Available Cases (QTY)",
            template="plotly_white",
            legend=dict(title="Category", orientation="h", y=1.1, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    ## Low Stock Insights: A Deeper Dive

    This analysis focuses on products with low stock levels. The first chart breaks down these items by category, allowing you to quickly pinpoint areas of concern. The second chart visualizes the relationship between wholesale price and available quantity, offering a more granular perspective on inventory levels for each product.  
    """)

def create_profit_margin_analysis_plot(df):
    df["Profit Margin"] = df["Retail price"] - df["Wholesale price"]
    df_sorted = df.sort_values(by="Profit Margin", ascending=False)

    fig = go.Figure(go.Bar(
        x=df_sorted['Product name'],
        y=df_sorted['Profit Margin'],
        marker=dict(color=df_sorted['Profit Margin'], colorscale='teal'),
        text=df_sorted['Profit Margin'].apply(lambda x: f'${x:,.2f}'),
        hovertemplate="<b>%{x}</b><br>Profit Margin: %{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Profit Margins: Low Stock Items",
        xaxis_title="Product Name",
        yaxis_title="Profit Margin",
        xaxis_tickangle=45,
        xaxis={'categoryorder':'total descending'},
        template="plotly_white",
        legend=dict(title="Profit Margin", orientation="h", y=1.1, x=0.5, xanchor='center')
    )
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Profitability Focus: Low Stock Items

    This bar chart highlights the profit margins of your low-stock items, sorted from highest to lowest. Prioritize replenishing high-margin products to maximize potential revenue and avoid stockouts.  
    """)

def create_low_stock_by_manufacturer_bar_plot(df):
    low_stock_counts = df.groupby("Manufacturer name")["Product name"].count().reset_index()

    fig = go.Figure(go.Bar(
        x=low_stock_counts['Manufacturer name'],
        y=low_stock_counts['Product name'],
        marker=dict(color=low_stock_counts['Product name']),
                    #colorscale='Pastel'),
        text=low_stock_counts['Product name'],
        hovertemplate="<b>%{x}</b><br>Number of Low Stock Items: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Low Stock Items by Manufacturer",
        xaxis_title="Manufacturer",
        yaxis_title="Number of Low Stock Items",
        xaxis_tickangle=45,
        xaxis={'categoryorder': 'total descending'},
        template="plotly_white",
        legend=dict(title="Manufacturer", orientation="h", y=1.1, x=0.5, xanchor='center')
    ) 
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Low Stock Breakdown: Manufacturer Focus

    This bar chart highlights the manufacturers with the highest number of low-stock items, providing insights into potential supplier-related challenges or product popularity. By analyzing this breakdown, you can proactively address inventory concerns and strengthen your supply chain relationships.  
    """)


#Analyzing the correlation Between Price and Available Quantity
def create_interactive_price_vs_quantity_plot(df):
    df['Wholesale price'] = pd.to_numeric(df['Wholesale price'], errors='coerce')

    fig = px.scatter(
        df, 
        x="Wholesale price", 
        y="Available cases (QTY)", 
        trendline="ols",
        title="Price vs. Quantity: Low-Stock Items",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        hovertemplate="<b>Product:</b> %{text}<br><b>Wholesale Price:</b> %{x}<br><b>Available Cases (QTY):</b> %{y}<extra></extra>",
        text=df['Product name']
    )
    
    fig.update_layout(
        xaxis_title="Wholesale Price", 
        yaxis_title="Available Cases (QTY)",
        legend=dict(title="Category", orientation="h", y=1.1, x=0.5, xanchor='center')
    )
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Price vs. Quantity: Low-Stock Item Analysis

    This scatter plot explores the relationship between wholesale price and available quantity for products currently low in stock. The trendline helps you visualize the general association between these factors. Analyze this visualization to inform your inventory management decisions and potentially predict future stock requirements.
    """)

def create_quantity_price_ratio_plot(df):
    df['Retail price'] = pd.to_numeric(df['Retail price'], errors='coerce')
    df["QTY/Price Ratio"] = df["Available cases (QTY)"] / df["Retail price"]
    df_sorted = df.sort_values(by="QTY/Price Ratio")

    fig = px.bar(
        df_sorted, 
        y='Product name', 
        x='QTY/Price Ratio', 
        color='QTY/Price Ratio', 
        orientation='h',
        title="Quantity/Price Ratio: Low-Stock Items",
        color_continuous_scale='purples', 
        text='QTY/Price Ratio',
        template="plotly_white"
    )
    
    fig.update_traces(
        texttemplate='%{text:.2f}', 
        textposition='outside',
        hovertemplate="<b>Product:</b> %{y}<br><b>QTY/Price Ratio:</b> %{x:.2f}<extra></extra>"
    )
    
    fig.update_layout(
        xaxis_title="QTY/Price Ratio", 
        yaxis_title="Product Name",
        legend=dict(title="Category", orientation="h", y=1.1, x=0.5, xanchor='center')
    )
    
    st.plotly_chart(fig)

    st.markdown("""
    ## Quantity/Price Ratio: A Closer Look at Low Stock

    This horizontal bar chart visualizes the ratio of available quantity to retail price for each low-stock item. Products with higher ratios might indicate overstocking or potential pricing issues, while those with lower ratios could signal high demand or potential stock shortages.
    """)