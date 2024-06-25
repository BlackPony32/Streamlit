import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# from utils import plotly_preproc

#is used to call instead of st.plotly_chart(fig)
def plotly_preproc(fig):
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)

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

#Total sales
def visualize_product_analysis(data, product_col='Product name', grand_total_col='Grand total'):
    product_data = data.groupby(product_col)[grand_total_col].agg(['sum', 'count']).sort_values(by='sum', ascending=False)

    tab1, tab2 = st.tabs(["Total Sales", "Order Distribution"])

    with tab1:
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=product_data.index,
            values=product_data['count'],
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Orders: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title_text='Order Distribution',
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        
        st.plotly_chart(fig)
        st.markdown("""
## Total Sales by Category

This chart shows how total sales are split across product categories over time. See which categories drive sales in each period and spot any trends.
""")

    with tab2:
        # Bar chart
        fig = go.Figure(data=[go.Bar(
            x=product_data.index,
            y=product_data['count'],
            marker=dict(
                color=product_data['count'],
                colorscale='Cividis'
            ),
            hovertemplate='<b>%{x}</b><br>Orders: %{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title_text="Distribution of Orders by Product",
            xaxis_title="Product",
            yaxis_title="Number of Orders",
            xaxis_tickangle=45,
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        
        st.plotly_chart(fig)
        st.markdown("""
## Order Distribution by Product

This chart displays the number of orders for each product, indicating their popularity and helping you manage inventory effectively.
""")

#Sales amount for each client (top 10)
def visualize_sales_trends(data, customer_col='Customer', product_col='Product name',
                           grand_total_col='Grand total', qty_col='QTY'):
    # Calculate top 10 customers
    top_customers = data.groupby(customer_col)[grand_total_col].sum().nlargest(10)

    # Create bar chart
    fig = go.Figure(data=[go.Bar(
        x=top_customers.index,
        y=top_customers.values,
        marker=dict(
            color=top_customers.values,
            colorscale='Bluyl',
            colorbar=dict(title="Sales Amount")
        ),
        hovertemplate='<b>%{x}</b><br>Sales Amount: $%{y:,.2f}<extra></extra>'
    )])

    # Update layout
    fig.update_layout(
        title_text="Top 10 Customers by Sales Amount",
        xaxis_title="Customer",
        yaxis_title="Sales Amount",
        xaxis_tickangle=45,
        yaxis_autorange="reversed",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )

    # Display chart
    st.plotly_chart(fig)

    # Display markdown
    st.markdown("""
    ## Top Customer Insights
    This chart highlights your top 10 customers by sales revenue. Prioritize these key relationships to drive future sales and consider loyalty programs to encourage repeat business.
    """)
        


#Plot with coloring of points by product type
def visualize_combined_analysis(data, product_col='Product name',
                               grand_total_col='Grand total', qty_col='QTY',
                               delivery_status_col='Delivery status'):
    # Group data by product and quantity
    grouped_data = data.groupby([product_col, qty_col])[grand_total_col].sum().reset_index()

    # Create a list of unique products
    products = grouped_data[product_col].unique()

    # Create the figure
    fig = go.Figure()

    # Add a trace for each product
    for product in products:
        product_data = grouped_data[grouped_data[product_col] == product]
        fig.add_trace(go.Bar(
            x=product_data[qty_col],
            y=product_data[grand_total_col],
            name=product,
            text=grouped_data[grouped_data[product_col] == product],  # Add product names to bars
            textposition='auto',  # Position text automatically
            hovertemplate='<b>%{text}</b><br>Quantity: %{x}<br>Sales Amount: $%{y:,.2f}<extra></extra>',
            width=0.8  # Make bars thicker
        ))

    # Update the layout
    fig.update_layout(
        title="Dependence between Quantity and Amount (by Product)",
        xaxis_title="Quantity",
        yaxis_title="Sales Amount",
        barmode='stack',
        legend_title="Product",
        bargap=0.2,  # Adjust gap between bar groups
        bargroupgap=0.1  # Adjust gap between bars in a group
    )

    # Display the chart
    st.plotly_chart(fig)

    st.markdown("""
    ## Sales by Quantity and Product

    This chart shows how sales revenue varies with the quantity of products sold, broken down by product name. Analyze which products contribute the most at different quantity levels.
    """)
#Analyzes discounts
def analyze_discounts(data):
    discount_counts = data["Discount type"].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=discount_counts.index,
        values=discount_counts.values,
        hole=0.4,
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        marker=dict(colors=px.colors.qualitative.Set1),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title_text="Distribution of Discount Types",
        showlegend=False
    )

    st.plotly_chart(fig)
    st.markdown("""
## Discount Usage

This chart presents the distribution of discount types used in orders. It highlights the proportion of orders associated with each discount category.
""")

def area_visualisation(data):
    columns = ['Grand total', 'Manufacturer specific discount', 'Customer discount']
    
    fig = go.Figure()
    
    for column in columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            stackgroup='one',
            name=column,
            hovertemplate='Index: %{x}<br>Amount: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Sales, Manufacturer, and Customer Discounts Over Time",
        xaxis_title="Index",
        yaxis_title="Amount",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig)
    st.markdown("""
    ## Sales, Manufacturer, and Customer Discounts Over Time

    This area chart displays how the grand total, manufacturer discounts, and customer discounts have fluctuated over time. Track how these values change, identifying periods of high discounts and understanding the overall impact of discounts on sales revenue. 
    """)