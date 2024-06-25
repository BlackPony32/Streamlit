import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np
from plotly.colors import sequential
from plotly.colors import qualitative
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
#Analyzes and visualizes the total inventory value by category
def df_analyze_inventory_value_by_category(df):
    df["Inventory Value"] = df["Available cases (QTY)"] * df["Wholesale price"]
    category_value = df.groupby("Category name")["Inventory Value"].sum().reset_index()

    fig = go.Figure(data=[go.Bar(
        x=category_value['Category name'],
        y=category_value['Inventory Value'],
        marker_color=px.colors.qualitative.Plotly[:len(category_value)],
        hovertemplate='<b>%{x}</b><br>Inventory Value: $%{y:,.2f}<extra></extra>'
    )])

    fig.update_layout(
        title="Inventory Value Distribution by Category",
        xaxis_title="Category",
        yaxis_title="Inventory Value",
        showlegend=False,
        legend_title_text='Category'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Inventory Value: A Category Breakdown

    This bar chart illustrates the proportional distribution of inventory value across different product categories, allowing you to quickly see which categories hold the most significant value within your inventory.
    """)

def df_analyze_quantity_vs_retail_price(df):
    for col in ["Retail price", "Wholesale price"]:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('$ ', ''))

    df['Wholesale price'] = df['Wholesale price'].fillna(0)

    categories = df["Category name"].unique()
    colors = px.colors.qualitative.Plotly[:len(categories)]
    color_map = dict(zip(categories, colors))

    fig = go.Figure()

    for category in categories:
        df_cat = df[df["Category name"] == category]
        fig.add_trace(go.Scatter(
            x=df_cat["Available cases (QTY)"],
            y=df_cat["Retail price"],
            mode='markers',
            marker=dict(
                size=df_cat["Wholesale price"],
                sizemode='area',
                sizeref=2.*max(df["Wholesale price"])/(40.**2),
                color=color_map[category]
            ),
            name=category,
            hovertemplate='<b>%{text}</b><br>Available Cases: %{x}<br>Retail Price: $%{y:.2f}<br>Wholesale Price: $%{marker.size:.2f}<extra></extra>',
            text=df_cat["Category name"]
        ))

    fig.update_layout(
        title="Quantity, Price, and Category: A Multi-Factor View",
        xaxis_title="Available Cases",
        yaxis_title="Retail Price",
        template="plotly_white",
        showlegend=True,
        legend_title_text='Category'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Quantity, Price, and Category: A Multi-Factor View

    This scatter plot provides a visual analysis of the interplay between available quantity, retail price, category, and wholesale price. Explore how these factors relate to each other, uncover potential trends within categories, and identify outliers that might require further investigation. Use these insights to inform your pricing, inventory, and product strategies. 
    """)

#Analyzing Inventory Value Distribution Across Manufacturers
def df_analyze_inventory_value_by_manufacturer(df):
    if df['Wholesale price'].dtype == 'object':
        df['Wholesale price'] = pd.to_numeric(df['Wholesale price'].str.replace(',', '').str.replace('$ ', ''))
    
    df["Inventory Value"] = df["Available cases (QTY)"] * df["Wholesale price"]
    manufacturer_value = df.groupby("Manufacturer name")["Inventory Value"].sum().reset_index()

    fig = go.Figure(data=[go.Bar(
        x=manufacturer_value['Manufacturer name'],
        y=manufacturer_value['Inventory Value'],
        marker_color=px.colors.qualitative.Plotly[:len(manufacturer_value)],
        hovertemplate='<b>%{x}</b><br>Inventory Value: $%{y:,.2f}<extra></extra>'
    )])

    fig.update_layout(
        title="Inventory Value Distribution by Manufacturer",
        xaxis_title="Manufacturer",
        yaxis_title="Inventory Value",
        showlegend=False,
        legend_title_text='Manufacturer'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Inventory Value: A Manufacturer Breakdown

    This bar chart illustrates the proportional distribution of inventory value across different manufacturers. This allows you to see at a glance which manufacturers contribute the most to your overall inventory value. 
    """)

#Analyzes and visualizes the average inventory value per unit for each product
def df_analyze_inventory_value_per_unit(df):
    # Ensure 'Wholesale price' is numeric
    if df['Wholesale price'].dtype == 'object':
        df['Wholesale price'] = pd.to_numeric(df['Wholesale price'].str.replace(',', '').str.replace('$ ', ''))
    
    # Calculate 'Inventory Value per Unit'
    df["Inventory Value per Unit"] = pd.to_numeric(df["Wholesale price"], errors='coerce')
    df = df.dropna(subset=["Inventory Value per Unit"])
    
    # Calculate total value per product
    df['Total Value'] = df["Inventory Value per Unit"] * df['Available cases (QTY)']
    
    # Group by 'Product name' and sum the 'Total Value'
    product_value = df.groupby("Product name")["Total Value"].sum().reset_index()

    # Create the bar chart using plotly.graph_objects
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=product_value['Product name'],
        y=product_value['Total Value'],
        marker_color=px.colors.qualitative.Pastel,
        hovertemplate=
        '<b>%{x}</b><br>' +
        'Total Value: $%{y:,.2f}<extra></extra>',
        name='Total Value'
    ))
    
    # Update the layout for better visualization
    fig.update_layout(
        title="Inventory Value Distribution by Product",
        xaxis_title="Product",
        yaxis_title="Total Value",
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Inventory Value: A Product Breakdown

    This bar chart breaks down the distribution of inventory value across individual products, providing insights into which products contribute the most to the overall inventory value.
    """)

# Comparing Average Retail Prices Across Categories
def df_compare_average_retail_prices(df):
    if df['Retail price'].dtype == 'object':
        df['Retail price'] = pd.to_numeric(df['Retail price'].str.replace(',', '').str.replace('$ ', ''))
    average_prices = df.groupby("Category name")["Retail price"].mean()
    
    # Create the pie chart using plotly.graph_objects
    fig = go.Figure(data=[go.Pie(
        values=average_prices.values,
        labels=average_prices.index,
        hole=0.3,  # Create a donut chart
        marker=dict(colors=px.colors.qualitative.Vivid),
        textinfo='percent+label',
        hovertemplate=
        '<b>%{label}</b><br>' +
        'Average Retail Price: $%{value:,.2f}<extra></extra>'
    )])

    fig.update_layout(
        title="Average Retail Price Distribution by Category",
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Average Retail Prices: A Category View

    This donut chart provides a visual representation of how average retail prices are distributed across different product categories. Easily compare the proportions and identify categories with higher or lower average prices. 
    """)
