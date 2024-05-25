import streamlit as st
import pandas as pd
import numpy as np

# import visualization package
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px


@st.cache_data  # this is for local host used
# @st.cache  # -> this is for deploy used
def load_data(data):
    df = pd.read_csv(data)
    df = df.iloc[:, 1:]
    return df


def run_eda_app():
    st.subheader("From Exploratory Data Analysis")
    df = load_data("cleancp.csv")
    # st.dataframe(df)
    # st.dataframe(df)

    submenu = st.sidebar.selectbox("SubMenu", ["Description", "Plots"])
    if submenu == "Description":
        st.dataframe(df)

        with st.expander("Descriptive Summary"):
            st.dataframe(df.describe())

        with st.expander("Top Manufacturer"):
            st.dataframe(df["manufacturer"].value_counts())

        with st.expander("Top Cars Category"):
            st.dataframe(df["category"].value_counts())

        with st.expander("Top Model Car"):
            st.dataframe(df["model"].value_counts())

    elif submenu == "Plots":
        st.subheader("Plots")

        # layouts
        col1, col2 = st.columns([2, 1])

        with col1:
            with st.expander("Dist Plot of Top Manufacturer"):
                # fig = plt.figure()
                # sns.countplot(x=df['gender'])
                # st.pyplot(fig)

                man_df = df["manufacturer"].value_counts().to_frame()
                man_df = man_df.reset_index()
                man_df.columns = ["Manufacturer Type", "Counts"]
                # st.dataframe(gen_df)

                p1 = px.pie(man_df, names="Manufacturer Type", values="Counts")
                st.plotly_chart(p1, use_container_width=True)

        with col2:
            with st.expander("Top Cars Category"):
                st.dataframe(df["category"].value_counts().to_frame())

        with st.expander("Frequency of Car Atributes"):
            p2 = px.bar(
                df,
                x=df["model"].value_counts().index,
                y=df["model"].value_counts().values,
            )
            st.plotly_chart(p2)
            p3 = px.bar(
                df,
                x=df["category"].value_counts().index,
                y=df["category"].value_counts().values,
            )
            st.plotly_chart(p3)

        with st.expander("Correlation Plot"):
            columns = df.select_dtypes(include=["int64", "float64"]).columns.to_list()
            corr_matrix = df[columns].corr()
            fig = plt.figure(figsize=(20, 10))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="crest",
                linewidth=0.5,
                annot_kws={"size": 15},
            )
            st.pyplot(fig)
