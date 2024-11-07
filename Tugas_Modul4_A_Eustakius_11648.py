import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px  #digunakan untuk membuat grafik scatter 3D
import numpy as np
from sklearn.metrics import pairwise_distances  #menghitung jarak antar data
import plotly.graph_objects as go  #digunakan untuk membuat grafik

#fungsi untuk membuat scatter plot 3D dan menentukan cluster dari titik baru
def scatter(model, model_name, data, new_point, features, color_scale, title):
    clusters = model.fit_predict(data[features])  #untuk memprediksi cluster untuk setiap titik pada data menggunakan model
    data[f"{model_name}_Cluster"] = clusters

    #menentukan cluster untuk titik baru
    if model_name == "KMeans_model":
        #pada k-means, dilakukan prediksi Langsung
        new_cluster = model.predict(new_point[features])[0]
    else:
        #pada agglomerative dan dbscan dilakukan dengan cara menghitung jarak titik terdekat
        distances = pairwise_distances(new_point[features], data[features])
        nearest_index = distances.argmin()
        new_cluster = clusters[nearest_index]

    #membuat grafik 3D menggunakan Plotly Express
    fig = px.scatter_3d(data, x='Avg_Credit_Limit', y='Total_Credit_Cards', z='Total_visits_online',
                        color=f"{model_name}_Cluster", title=title, color_continuous_scale=color_scale)

    #menambahkan titik baru pada grafik
    fig.add_trace(
        go.Scatter3d(
            x=new_point['Avg_Credit_Limit'],
            y=new_point['Total_Credit_Cards'],
            z=new_point['Total_visits_online'],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='New Point'
        )
    )

    return fig, new_cluster

st.set_page_config(
    page_title="220711648 - Unsupervised Learning",  # Diisi dengan NPM Anda
    page_icon="📊",  # Icon
    layout="wide",
    initial_sidebar_state="expanded",
)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("<h1 style='text-align: center;'>Unsupervised Learning - 220711648</h1>", unsafe_allow_html=True)
    st.dataframe(input_data)

    #direktori tempat penyimpanan ketiga model yang telah di dump sebelumnya
    model_directory = r'D:\ATMAJAYA\Semester5\PMDPL\Unsupervised Learning (Praktek)\Tugas4_A_11648'
    model_path = {
        "AGG_model": os.path.join(model_directory, r'AGG_model.pkl'),
        "KMeans_model": os.path.join(model_directory, r'KMeans_model.pkl'),
        "DBSCAN_model": os.path.join(model_directory, r'DBSCAN_model.pkl'),
    }

    #load model ketiga model ke dalam dictionary
    models = {}
    for model_name, path in model_path.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            st.write(f"Model {model_name} tidak ditemukan di path : ", path)

    #sidebar untuk memasukkan nilai untuk titik baru yang akan diprediksi clusternya
    avg_CL = st.sidebar.number_input("Average Credit Limit", 0, 100000)
    sum_CC = st.sidebar.number_input("Total Credit Cards", 0, 10)
    sum_VO = st.sidebar.number_input("Total Visits Online", 0, 16)

    if st.sidebar.button("Prediksi !"):
        #fitur yang digunakan untuk memprediksi
        features = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_online']
        #memasukkan data titik baru ke dalam DataFrame
        new_point = pd.DataFrame({
            'Avg_Credit_Limit': [avg_CL],
            'Total_Credit_Cards': [sum_CC],
            'Total_visits_online': [sum_VO]
        })

        #model clustering yang digunakan dan warna grafik scatternya
        #warna dapat diubah sesuai keinginan
        cluster_method = [
            ("KMeans_model", models["KMeans_model"], "KMeans Clustering", px.colors.sequential.Cividis),
            ("AGG_model", models["AGG_model"], "Agglomerative Clustering", px.colors.sequential.Mint),
            ("DBSCAN_model", models["DBSCAN_model"], "DBSCAN Clustering", px.colors.sequential.Plasma)
        ]

        #membuat tiga kolom untuk menampilkan grafik
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, (model_name, model, title, color_scale) in enumerate(cluster_method):
            fig, new_cluster = scatter(model, model_name, input_data, new_point, features, color_scale, title)

            with cols[i]:
                st.plotly_chart(fig)
                st.markdown(f"<p style='text-align: center;'>Titik Dari Data yang baru masuk ke dalam cluster : {new_cluster}</p>", unsafe_allow_html=True)
