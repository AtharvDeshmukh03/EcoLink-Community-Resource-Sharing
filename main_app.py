import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss
from catboost import CatBoostClassifier
import random

# --------------------------
# File paths
# --------------------------
RESOURCES_FILE = 'resources.csv'
REQUESTS_FILE = 'requests.csv'
MODEL_FILE = 'catboost_resource_model.cbm'

# --------------------------
# Resource Categories
# --------------------------
CATEGORY_OPTIONS = [
    "Tools", "Sports", "Electronics", "Books", "Kitchen",
    "Furniture", "Vehicles", "Musical Instruments", "Clothing",
    "Gardening", "Toys", "Appliances", "Outdoor", "Other"
]

# --------------------------
# Initialize CSV files
# --------------------------
def initialize_csv(file_path, columns):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)

initialize_csv(RESOURCES_FILE, ['id', 'title', 'category', 'description', 'location',
                                'availability_start', 'availability_end', 'condition', 'rating'])
initialize_csv(REQUESTS_FILE, ['request_id', 'user_name', 'resource_id', 'resource_title',
                               'start_date', 'end_date', 'status'])

# --------------------------
# Load data
# --------------------------
resources_df = pd.read_csv(RESOURCES_FILE)
requests_df = pd.read_csv(REQUESTS_FILE)

# --------------------------
# Validate dates
# --------------------------
def validate_dates(start_date, end_date):
    today = datetime.today().date()
    if start_date > end_date:
        return False, "Start date must be before or equal to end date."
    if start_date < today or end_date < today:
        return False, "Dates cannot be in the past."
    return True, ""

# --------------------------
# Check availability
# --------------------------
def is_resource_available(resource_id, start_date, end_date, requests_df):
    existing = requests_df[
        (requests_df['resource_id'] == resource_id) &
        (requests_df['status'] == 'Confirmed')
    ]
    for _, row in existing.iterrows():
        existing_start = datetime.strptime(row['start_date'], "%Y-%m-%d").date()
        existing_end = datetime.strptime(row['end_date'], "%Y-%m-%d").date()
        if (start_date <= existing_end) and (end_date >= existing_start):
            return False
    return True

# --------------------------
# Add request
# --------------------------
def add_request(user_name, resource_id, resource_title, start_date, end_date, status):
    global requests_df, resources_df
    request_id = 1 if requests_df.empty else requests_df['request_id'].max() + 1
    new_request = {
        'request_id': request_id,
        'user_name': user_name,
        'resource_id': resource_id,
        'resource_title': resource_title,
        'start_date': start_date.strftime("%Y-%m-%d"),
        'end_date': end_date.strftime("%Y-%m-%d"),
        'status': status
    }
    requests_df = pd.concat([requests_df, pd.DataFrame([new_request])], ignore_index=True)
    requests_df.to_csv(REQUESTS_FILE, index=False)

    if status == "Confirmed":
        resources_df.loc[resources_df['id'] == resource_id, 'availability_start'] = ""
        resources_df.loc[resources_df['id'] == resource_id, 'availability_end'] = ""
        resources_df.to_csv(RESOURCES_FILE, index=False)

# --------------------------
# Build FAISS Index
# --------------------------
@st.cache_resource
def build_index(df):
    TEXT_COLS = ["title", "location", "condition", "description"]
    df["text"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return model, index

# --------------------------
# Sidebar navigation
# --------------------------
st.sidebar.title("Community Resource Sharing")
page = st.sidebar.radio("Navigate", ["Home", "Smart Search", "Offer Resource", "My Dashboard", "ML Predictions"])

# --------------------------
# Home Page
# --------------------------
if page == "Home":
    st.title("🏠 Welcome to the Community Resource Hub!")

    # Quick Stats
    st.subheader("📊 Quick Stats")
    total_resources = len(resources_df)
    total_requests = len(requests_df)
    confirmed_requests = len(requests_df[requests_df['status'] == 'confirmed'])
    waitlist_requests = len(requests_df[requests_df['status'] == 'requested'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resources", total_resources)
    col2.metric("Total Requests", total_requests)
    col3.metric("Confirmed Bookings", confirmed_requests)
    col4.metric("Waitlist Requests", waitlist_requests)

    # Featured Resources (random 5)
    st.subheader("🌟 Featured Resources")
    available_resources = resources_df[
        (resources_df['availability_start'] != "") & (resources_df['availability_end'] != "")
    ]
    if not available_resources.empty:
        sample_resources = available_resources.sample(min(5, len(available_resources)))
        for _, row in sample_resources.iterrows():
            with st.expander(f"{row['title']} ({row['category']})"):
                st.write(f"📍 Location: {row['location']}")
                st.write(f"📅 Available: {row['availability_start']} to {row['availability_end']}")
                st.write(f"⭐ Condition: {row['condition']} | Rating: {row['rating']}")
                if st.button("Request Resource", key=f"home_req_{row['id']}"):
                    st.session_state['selected_resource'] = row['id']
                    st.session_state['request_active'] = True

    # Request form
    if st.session_state.get('request_active', False):
        resource_id = st.session_state['selected_resource']
        resource = resources_df.loc[resources_df['id'] == resource_id].squeeze()
        st.subheader("Request Resource Form")
        with st.form("request_form"):
            user_name = st.text_input("Your Name")
            start_date = st.date_input("Start Date", datetime.today())
            end_date = st.date_input("End Date", datetime.today())
            submitted = st.form_submit_button("Submit Request")

            if submitted:
                is_valid, msg = validate_dates(start_date, end_date)
                if not is_valid:
                    st.error(msg)
                elif user_name.strip() == "":
                    st.error("Please enter your name.")
                elif is_resource_available(resource_id, start_date, end_date, requests_df):
                    add_request(user_name, resource_id, resource['title'], start_date, end_date, "Confirmed")
                    st.success(f"Booking confirmed for {resource['title']}!")
                    st.session_state['request_active'] = False
                else:
                    add_request(user_name, resource_id, resource['title'], start_date, end_date, "Waitlist")
                    st.warning(f"The resource is already booked for these dates. You have been added to the waitlist.")
                    st.session_state['request_active'] = False

# --------------------------
# Smart Search Page
# --------------------------
elif page == "Smart Search":
    st.title("🔍 Smart Search in Resources")
    model, index = build_index(resources_df)
    st.success(f"Indexed {len(resources_df)} resources.")

    def search(query, k=5):
        q_vec = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(q_vec, k * 2)

        results_list = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist >= 0:
                row = resources_df.iloc[idx].copy()
                row["score"] = max(1 - dist / 2, 0)
                results_list.append(row)
            if len(results_list) >= k:
                break
        return pd.DataFrame(results_list)

    query = st.text_input("Enter your search query:")
    if query:
        results = search(query, k=5)
        if not results.empty:
            st.write("### Top Results")
            for _, row in results.iterrows():
                with st.expander(f"{row['title']} ({row.get('category','N/A')})"):
                    for col in ["title", "category", "description", "location", "condition", "rating", "score"]:
                        st.write(f"**{col.title()}:** {row.get(col, '')}")
                    if st.button("Request Resource", key=f"smart_req_{row['id']}"):
                        st.session_state['selected_resource'] = row['id']
                        st.session_state['request_active'] = True
        else:
            st.warning("No matching resources found.")

    if st.session_state.get('request_active', False):
        resource_id = st.session_state['selected_resource']
        resource = resources_df.loc[resources_df['id'] == resource_id].squeeze()
        st.subheader("Request Resource Form")
        with st.form("smart_request_form"):
            user_name = st.text_input("Your Name")
            start_date = st.date_input("Start Date", datetime.today())
            end_date = st.date_input("End Date", datetime.today())
            submitted = st.form_submit_button("Submit Request")

            if submitted:
                is_valid, msg = validate_dates(start_date, end_date)
                if not is_valid:
                    st.error(msg)
                elif user_name.strip() == "":
                    st.error("Please enter your name.")
                elif is_resource_available(resource_id, start_date, end_date, requests_df):
                    add_request(user_name, resource_id, resource['title'], start_date, end_date, "Confirmed")
                    st.success(f"Booking confirmed for {resource['title']}!")
                    st.session_state['request_active'] = False
                else:
                    add_request(user_name, resource_id, resource['title'], start_date, end_date, "Waitlist")
                    st.warning(f"The resource is already booked for these dates. You have been added to the waitlist.")
                    st.session_state['request_active'] = False

# --------------------------
# Offer Resource Page
# --------------------------
elif page == "Offer Resource":
    st.title("📦 Offer a New Resource")
    with st.form("offer_form"):
        title = st.text_input("Resource Title")
        category = st.selectbox("Category", CATEGORY_OPTIONS)
        description = st.text_area("Description")
        location = st.text_input("Location")
        start_date = st.date_input("Available From", datetime.today())
        end_date = st.date_input("Available Until", datetime.today())
        condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor"])
        rating = st.slider("Rating (0-5)", 0, 5, 3)
        submitted = st.form_submit_button("Offer Resource")

        if submitted:
            is_valid, msg = validate_dates(start_date, end_date)
            if not is_valid:
                st.error(msg)
            elif title.strip() == "" or location.strip() == "":
                st.error("Title and location are required.")
            else:
                new_id = 1 if resources_df.empty else resources_df['id'].max() + 1
                new_resource = {
                    'id': new_id,
                    'title': title,
                    'category': category,
                    'description': description,
                    'location': location,
                    'availability_start': start_date.strftime("%Y-%m-%d"),
                    'availability_end': end_date.strftime("%Y-%m-%d"),
                    'condition': condition,
                    'rating': rating
                }
                resources_df = pd.concat([resources_df, pd.DataFrame([new_resource])], ignore_index=True)
                resources_df.to_csv(RESOURCES_FILE, index=False)

                # Clear cache so Smart Search reindexes
                build_index.clear()
                st.success(f"{title} has been added to the community pool!")

# --------------------------
# Admin Dashboard
# --------------------------
elif page == "My Dashboard":
    st.title("🛠 Admin Dashboard")

    st.subheader("📊 Summary Metrics")
    total_requests = len(requests_df)
    confirmed = len(requests_df[requests_df['status'] == 'confirmed'])
    waitlist = len(requests_df[requests_df['status'] == 'requested'])
    available_resources = len(resources_df[(resources_df['availability_start'] != "") & (resources_df['availability_end'] != "")])
    booked_resources = len(resources_df) - available_resources

    st.metric("Total Requests", total_requests)
    st.metric("Confirmed Bookings", confirmed)
    st.metric("Pending Requests (Waitlist)", waitlist)
    st.metric("Available Resources", available_resources)
    st.metric("Booked Resources", booked_resources)

    st.subheader("📂 Grouped Requests Overview")
    grouped = requests_df.groupby('status')
    for status, group in grouped:
        st.write(f"**{status} Requests:** {len(group)}")
        if st.button(f"Show {status} Requests Details"):
            st.dataframe(group[['request_id', 'user_name', 'resource_title', 'start_date', 'end_date']])

    st.subheader("📈 Analytics & Trends")
    popular_resources = requests_df.groupby('resource_title')['request_id'].count().reset_index().sort_values(by='request_id', ascending=False)
    if not popular_resources.empty:
        st.write("### 🔥 Most Popular Resources")
        fig = px.bar(popular_resources.head(10), x='resource_title', y='request_id',
                     labels={'resource_title': 'Resource', 'request_id': 'Requests'},
                     title="Top 10 Popular Resources")
        st.plotly_chart(fig)

    active_users = requests_df.groupby('user_name')['request_id'].count().reset_index().sort_values(by='request_id', ascending=False)
    if not active_users.empty:
        st.write("### 👥 Most Active Users")
        fig = px.bar(active_users.head(10), x='user_name', y='request_id',
                     labels={'user_name': 'User', 'request_id': 'Requests'},
                     title="Top 10 Active Users")
        st.plotly_chart(fig)

    if not requests_df.empty:
        requests_df['start_date'] = pd.to_datetime(requests_df['start_date'])
        trend = requests_df.groupby(requests_df['start_date'].dt.to_period('M')).size().reset_index(name='counts')
        trend['start_date'] = trend['start_date'].dt.to_timestamp()
        fig = px.line(trend, x='start_date', y='counts', markers=True, title="Request Trends Over Time")
        st.plotly_chart(fig)

# --------------------------
# ML Predictions
# --------------------------
elif page == "ML Predictions":
    st.title("🤖 ML Predictions using CatBoost")

    @st.cache_resource
    def load_model():
        model = CatBoostClassifier()
        model.load_model(MODEL_FILE)
        return model

    try:
        model = load_model()
        resources = pd.read_csv(RESOURCES_FILE)

        resources["availability_start"] = pd.to_datetime(resources["availability_start"], errors="coerce")
        resources["availability_end"] = pd.to_datetime(resources["availability_end"], errors="coerce")
        resources["availability_duration"] = (resources["availability_end"] - resources["availability_start"]).dt.days
        resources["availability_duration"] = resources["availability_duration"].clip(0, None)
        resources["rating"] = resources["rating"].clip(0, 5)
        resources["category_condition"] = resources["category"].astype(str) + "_" + resources["condition"].astype(str)

        features = ["category", "condition", "rating", "availability_duration", "category_condition"]

        resources["request_probability"] = model.predict_proba(resources[features])[:, 1]
        sorted_resources = resources[["title", "request_probability"]].sort_values(by="request_probability", ascending=False)
        top_10 = sorted_resources.head(10)

        st.subheader("📊 Top Predicted Resources")
        st.dataframe(top_10.reset_index(drop=True))

        sampled_top = top_10.iloc[::2]
        fig = px.bar(
            sampled_top,
            x="title",
            y="request_probability",
            title="Top Predicted Resources",
            labels={"title": "Resource", "request_probability": "Probability"},
            range_y=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load model or compute predictions: {e}")
