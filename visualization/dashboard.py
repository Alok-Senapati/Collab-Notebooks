import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configure page
st.set_page_config(page_title="E-commerce Dashboard", layout="wide")

# Title
st.title("📊 E-commerce Performance Dashboard")

# Load data (recreate the e-commerce dataset)
np.random.seed(42)
n = 40
categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 'Sports']
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata']

ecommerce = pd.DataFrame({
    'order_id': [f'ORD{str(i).zfill(3)}' for i in range(1, n + 1)],
    'order_date': pd.date_range('2025-01-01', periods=n, freq='3D'),
    'product_category': np.random.choice(categories, n),
    'city': np.random.choice(cities, n),
    'quantity': np.random.randint(1, 5, n),
    'unit_price': np.random.randint(500, 50000, n),
    'customer_rating': np.round(np.random.uniform(2.5, 5.0, n), 1)
})
ecommerce['total_amount'] = ecommerce['unit_price'] * ecommerce['quantity']

# Sidebar filters
st.sidebar.header("Filters")
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=categories,
    default=categories
)

selected_cities = st.sidebar.multiselect(
    "City",
    options=cities,
    default=cities
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(ecommerce['order_date'].min(), ecommerce['order_date'].max())
)

# Filter data
filtered_df = ecommerce[
    (ecommerce['product_category'].isin(selected_categories)) &
    (ecommerce['city'].isin(selected_cities)) &
    (ecommerce['order_date'] >= pd.to_datetime(date_range[0])) &
    (ecommerce['order_date'] <= pd.to_datetime(date_range[1]))
]

# KPI Metrics (Summary)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Revenue", f"₹{filtered_df['total_amount'].sum():,.0f}")

with col2:
    st.metric("Total Orders", f"{len(filtered_df)}")

with col3:
    st.metric("Avg Order Value", f"₹{filtered_df['total_amount'].mean():,.0f}")

with col4:
    st.metric("Avg Rating", f"{filtered_df['customer_rating'].mean():.1f} / 5.0")

# Trend Chart
st.subheader("📈 Monthly Revenue Trend")
monthly_revenue = filtered_df.groupby(filtered_df['order_date'].dt.to_period('M'))['total_amount'].sum().reset_index()
monthly_revenue['order_date'] = monthly_revenue['order_date'].astype(str)

fig_trend = px.line(
    monthly_revenue,
    x='order_date',
    y='total_amount',
    title='Revenue Over Time',
    labels={'order_date': 'Month', 'total_amount': 'Revenue (INR)'},
    markers=True
)
st.plotly_chart(fig_trend, use_container_width=True)

# Breakdown Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Revenue by Category")
    category_revenue = filtered_df.groupby('product_category')['total_amount'].sum().sort_values()
    fig_category = px.bar(
        category_revenue,
        x=category_revenue.values,
        y=category_revenue.index,
        orientation='h',
        title='Revenue by Product Category',
        labels={'x': 'Revenue (INR)', 'y': 'Category'}
    )
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    st.subheader("🔍 Price vs Rating")
    fig_scatter = px.scatter(
        filtered_df,
        x='unit_price',
        y='customer_rating',
        size='quantity',
        color='product_category',
        hover_data=['city', 'total_amount'],
        title='Unit Price vs Customer Rating',
        labels={'unit_price': 'Unit Price (INR)', 'customer_rating': 'Rating'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Detail: Raw Data Table
st.subheader("📋 Filtered Data")
st.dataframe(filtered_df, use_container_width=True)
