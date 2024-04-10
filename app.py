import streamlit as st
import numpy as np
np.random.seed(42)
from src.utils.reco_update import RecoUpdate
from src.configs.config import PathsConfig

path = PathsConfig(isCurrentIsRoot=True)
catalog = RecoUpdate(isCurrentIsRoot=True)

st.title('Recommendation System')

def print_items(recommended_items, i):
    st.image(str(path.sample_picdict[recommended_items['type'].iloc[i]]), caption=recommended_items['title'].iloc[i])
    st.write(recommended_items['price'].iloc[i])
    st.write(recommended_items['quantity'].iloc[i])

search_query = st.text_input('Search for items (cleaning, drinks, chocolates):')

if st.button('Submit'):
    recommended_items = catalog.recommend_catalog(query_string=search_query)

    # Display images in a row
    col1, col2, col3 = st.columns(3)  # Create three columns
    with col1:
        print_items(recommended_items, 0)
    with col2:
        print_items(recommended_items, 1)
    with col3:
        print_items(recommended_items, 2)

# Add a title to the sidebar
st.sidebar.image(str(path.sample_cooppic))
st.sidebar.title("Realtime Update Catalog")

# Add input fields to the sidebar
title = st.sidebar.text_input("Title", "")
type_options = ['drink', 'choco', 'clean']
selected_type = st.sidebar.selectbox("Type", options=type_options)
quantity = st.sidebar.text_input("Quantity", "")
price = st.sidebar.number_input("Price", value=0.0)

# Add "Update Catalog" button
if st.sidebar.button("Update Catalog"):
    catalog.update_catalog(title, selected_type, quantity, price,)