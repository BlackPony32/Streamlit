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

#_________________Sales Trends Function  (with Plotly)_______________________________
def visualize_sales_trends(data, customer_col='Customer', product_col='Product name', 
                           grand_total_col='Grand total', qty_col='QTY'):
    tab1, tab2 = st.tabs(["Top Customers", "Monthly Trend"])

    with tab1:
        st.subheader("Total Sales by Customer (Top 10)")
        top_customers = data.groupby(customer_col)[grand_total_col].sum().nlargest(10)
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_customers.index,
            y=top_customers.values,
            marker=dict(
                color=top_customers.values,
                colorscale='Bluyl'
            ),
            hovertemplate='<b>Customer:</b> %{x}<br><b>Sales Amount:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Update the layout
        fig.update_layout(
            title="Top 10 Customers by Sales Amount",
            xaxis_tickangle=45,
            yaxis_title="Sales Amount",
            xaxis_title="Customer",
            coloraxis_colorbar=dict(title="Sales Amount")
        )
        
        # Display the chart in Streamlit
        st.plotly_chart(fig)

        st.markdown("""
        ## Top Customer Insights

        This chart highlights your top 10 customers by sales amount. Use it to:

        * **Focus on High-Value Customers:** Prioritize relationships with clients who contribute the most to revenue.
        * **Evaluate Sales Performance:** Compare sales contributions and identify potential gaps or disparities.
        * **Segment and Target Effectively:** Group similar customers for targeted marketing and tailored offerings.
        """)

    with tab2:
        st.subheader("Monthly Sales Trend")
        data['Created at'] = pd.to_datetime(data['Created at'])

        # Specify the column name for grand total
        grand_total_col = 'Grand total'

        # Calculate monthly sales
        monthly_sales = data.groupby(pd.Grouper(key='Created at', freq='M'))[grand_total_col].sum()

        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales.values,
            mode='lines+markers',
            name='Sales Amount',
            hovertemplate='<b>Month:</b> %{x}<br><b>Sales Amount:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Update the layout
        fig.update_layout(
            title="Monthly Sales Trend",
            xaxis_title="Month",
            yaxis_title="Sales Amount",
            hovermode='x unified'
        )

        # Update the layout
        #fig.update_layout(xaxis_title="Month", yaxis_title="Sales Amount")
        st.plotly_chart(fig)
        st.markdown("""
        ## Monthly Sales Trend Insights

        This line chart tracks your overall sales trajectory over time. Use it to:

        * **Identify Sales Patterns:** Spot trends (increasing/decreasing sales), seasonality, and periods of growth or decline.
        * **Analyze Monthly Performance:** Investigate months with exceptionally high or low sales to understand contributing factors.
        * **Improve Sales Forecasting:** Forecast future sales based on observed trends and seasonality.
        """)

#_________________Product Analysis Function (with Plotly)___________________________
def visualize_product_analysis(data, product_col='Product name', grand_total_col='Grand total'):
    product_data = data.groupby(product_col)[grand_total_col].agg(['sum', 'count']).sort_values(by='sum', ascending=False)

    tab1, tab2 = st.tabs(["Total Sales", "Order Distribution"])

    with tab1:
        fig = go.Figure(data=[go.Pie(
            labels=product_data.index, 
            values=product_data['sum'],
            hovertemplate='<b>%{label}</b><br>Sales: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>',
            textinfo='percent+label'
        )])
        fig.update_layout(
            title="Total Sales by Product",
            colorway=px.colors.qualitative.Vivid,
            title_x=0.5
        )
        st.plotly_chart(fig)

        st.markdown("""
        ## Total Sales by Product

        This pie chart presents the share of total sales revenue generated by each product. It helps you:

        * **Identify Top Performers:** Quickly see which products contribute the most to overall revenue.
        * **Prioritize Product Focus:** Determine where to concentrate marketing and sales efforts for maximum impact.
        * **Analyze Performance Trends:** Track changes in product sales contributions over time. 
        """)

    with tab2:
        fig = go.Figure(data=[go.Bar(
            x=product_data.index, 
            y=product_data['count'],
            hovertemplate='<b>%{x}</b><br>Orders: %{y}<extra></extra>',
            marker_color=product_data['count'],
            marker_colorscale='Cividis'
        )])
        fig.update_layout(
            title="Distribution of Orders by Product",
            xaxis_title="Product",
            yaxis_title="Number of Orders",
            xaxis_tickangle=45,
            title_x=0.5
        )
        st.plotly_chart(fig)

        st.markdown("""
        ## Order Distribution by Product

        This bar chart shows the number of orders for each product, providing insights into:

        * **Product Popularity:** Identify products with high order volumes, indicating popularity or demand.
        * **Inventory Planning:** Use order volume to inform inventory management and prevent stock shortages for popular items. 
        * **Performance Comparison:** See which products have relatively low order numbers, which might suggest areas for improvement.
        """)

#_________________Discount Analysis Function (with Plotly)__________________________
def visualize_discount_analysis(data, discount_type_col='Discount type', total_discount_col='Total invoice discount'):
    """Visualizes discount analysis by type and top customers."""

    # Create tabs for different analyses
    tab1, tab2 = st.tabs(["Top Customers", "By Type"])

    with tab2:
        st.subheader("Discount Amount by Type")
        
        # Group by discount type and sum the total discounts
        discount_amounts = data.groupby(discount_type_col)[total_discount_col].sum().sort_values(ascending=False)
        
        # Create a pie chart for discount distribution by type
        fig = go.Figure(data=[go.Pie(
            labels=discount_amounts.index, 
            values=discount_amounts.values,
            hovertemplate='<b>%{label}</b><br>Discount Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>',
            textinfo='percent+label'
        )])
        fig.update_layout(
            title="Distribution of Discount Amount by Type",
            colorway=px.colors.qualitative.Set3,
            title_x=0.5
        )
        
        # Display the pie chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Provide markdown explanation
        st.markdown("""
        ## Discount Type Insights

        This pie chart shows the proportion of total discounts by type. It helps to:

        - **Evaluate Discount Strategy:** Identify which discount types are most impactful.
        - **Compare Discount Options:** Spot areas for optimization or experimentation.
        """)

    with tab1:
        st.subheader("Discount Amount by Customer (Top 10)")
        
        # Group by customer and sum the total discounts, then get the top 10 customers
        top_customers_discount = data.groupby('Customer')[total_discount_col].sum().nlargest(10)
        
        # Create a bar chart for top customers by discount amount
        fig = go.Figure(data=[go.Bar(
            x=top_customers_discount.index, 
            y=top_customers_discount.values,
            hovertemplate='<b>%{x}</b><br>Discount Amount: $%{y:,.2f}<extra></extra>',
            marker=dict(color=top_customers_discount.values, colorscale='Pinkyl')
        )])
        fig.update_layout(
            title="Discount Amount by Customer (Top 10)", 
            xaxis_tickangle=45, 
            yaxis_title="Discount Amount", 
            xaxis_title="Customer",
            title_x=0.5,
            coloraxis_showscale=False  # Hide the color scale
        )
        
        # Display the bar chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Provide markdown explanation
        st.markdown("""
        ## Top Discount Recipients 

        This chart shows the customers receiving the highest total discounts. It helps to:

        - **Align with Customer Strategies:** Ensure discounts align with business goals.
        - **Identify Negotiation Patterns:** Spot disparities or patterns in customer agreements.
        """)

    
# _________________Delivery Analysis Function (with Plotly)___________________________
def visualize_delivery_analysis(data, delivery_status_col='Delivery status', 
                                delivery_method_col='Delivery methods'):
    tab1, tab2 = st.tabs(["By Status", "By Method"])
    
    with tab1:
        st.subheader("Number of Orders by Delivery Status")
        delivery_status_counts = data[delivery_status_col].value_counts()
        
        # Create a pie chart for delivery status distribution
        fig = go.Figure(data=[go.Pie(
            labels=delivery_status_counts.index, 
            values=delivery_status_counts.values,
            hovertemplate='<b>%{label}</b><br>Orders: %{value}<br>Percentage: %{percent}<extra></extra>',
            textinfo='percent+label'
        )])
        fig.update_layout(
            title="Distribution of Orders by Delivery Status",
            colorway=px.colors.qualitative.Light24,
            title_x=0.5
        )
        
        # Display the pie chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ## Delivery Status Insights

        This pie chart displays the proportion of orders for each delivery status. It helps you:
        
        * **Monitor Delivery Efficiency:** Track the percentage of "Delivered" orders over time to assess overall efficiency.
        * **Spot Potential Issues:** A high percentage of "Cancelled" or "Returned" orders might indicate issues that require further investigation.
        """)

    with tab2:
        st.subheader("Number of Orders by Delivery Method")
        delivery_method_counts = data[delivery_method_col].value_counts()
        
        # Create a pie chart for delivery method distribution
        fig = go.Figure(data=[go.Pie(
            labels=delivery_method_counts.index, 
            values=delivery_method_counts.values,
            hovertemplate='<b>%{label}</b><br>Orders: %{value}<br>Percentage: %{percent}<extra></extra>',
            textinfo='percent+label'
        )])
        fig.update_layout(
            title="Distribution of Orders by Delivery Method",
            colorway=px.colors.qualitative.Bold,
            title_x=0.5
        )
        
        # Display the pie chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ## Delivery Method Insights

        This pie chart shows the distribution of orders across different delivery methods, helping you understand:

        * **Customer Preferences:** Which delivery methods are most popular among your customers?
        * **Operational Efficiency:** Are certain delivery methods used more frequently? This information can inform resource allocation and logistics planning. 
        """)

# _________________Payment Analysis Function (with Plotly)___________________________
def visualize_payment_analysis(data, payment_status_col='Payment status'):
    payment_status_counts = data[payment_status_col].value_counts()

    # Create a pie chart for payment status distribution
    fig = go.Figure(data=[go.Pie(
        labels=payment_status_counts.index, 
        values=payment_status_counts.values,
        hovertemplate='<b>%{label}</b><br>Orders: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='percent+label'
    )])
    fig.update_layout(
        title="Distribution of Orders by Payment Status",
        colorway=px.colors.qualitative.Pastel,
        title_x=0.5
    )

    # Display the pie chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ## Payment Status Insights

    This chart shows the distribution of orders by payment status. Use it to:

    * **Identify Payment Processing Issues:** Spot any anomalies in payment status distribution.
    * **Track Payment Trends:** Monitor changes in payment behavior over time.
    * **Gain Insights into Refund Frequency:** Understand how often refunds are processed.
    """)

# _________________Combined Analysis Function (with Plotly)___________________________
def visualize_combined_analysis(data, product_col='Product name', 
                               grand_total_col='Grand total', qty_col='QTY', 
                               delivery_status_col='Delivery status'):
    tab1, tab2 = st.tabs(["Quantity vs. Amount", "Orders by Product & Status"])
    
    with tab1:
        st.subheader("Relationship between Quantity and Amount (by Product)")
        scatter_data = [
            go.Scatter(
                x=data[data[product_col] == product][qty_col], 
                y=data[data[product_col] == product][grand_total_col],
                mode='markers',
                name=product,
                text=data[data[product_col] == product][product_col],
                hovertemplate='Quantity: %{x}<br>Sales Amount: %{y}<br>Product: %{text}<extra></extra>'
            ) for product in data[product_col].unique()
        ]
        fig = go.Figure(data=scatter_data)
        fig.update_layout(
            title="Relationship between Quantity and Amount (by Product)",
            xaxis_title="Quantity",
            yaxis_title="Sales Amount",
            xaxis_tickangle=45,
            template="plotly_white"
        )
        st.plotly_chart(fig)

        st.markdown("""
        ## Quantity vs. Amount Insights

        This plot reveals how quantity sold relates to sales amount. Analyze trends, outliers, and product performance to make informed pricing and inventory decisions.
        """)

    with tab2:
        st.subheader("Number of Orders by Product and Delivery Status")
        histogram_data = [
            go.Histogram(
                x=data[data[delivery_status_col] == status][product_col],
                name=status,
                marker=dict(line=dict(width=0.5)),
                hovertemplate='Product: %{x}<br>Number of Orders: %{y}<br>Delivery Status: %{text}<extra></extra>',
                text=data[data[delivery_status_col] == status][delivery_status_col]
            ) for status in data[delivery_status_col].unique()
        ]
        fig = go.Figure(data=histogram_data)
        fig.update_layout(
            title="Number of Orders by Product and Delivery Status",
            xaxis_title="Product",
            yaxis_title="Number of Orders",
            barmode='group',
            xaxis_tickangle=45,
            template="plotly_white"
        )
        st.plotly_chart(fig)

        st.markdown("""
        ## Orders and Fulfillment Insights

        This chart compares order volumes and fulfillment status. Identify potential bottlenecks, optimize inventory, and improve customer service by understanding product-level fulfillment patterns. 
        """)