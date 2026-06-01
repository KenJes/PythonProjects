import os
import streamlit as st
import sqlite3
import pandas as pd

# Conectar a la base
_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videogames.db")
conn = sqlite3.connect(_db_path)
df = pd.read_sql("SELECT * FROM vgsales_clean", conn)

# Dashboard
st.title("Ventas de Videojuegos")
st.bar_chart(df.groupby("Genre")["Global_Sales"].sum())
st.line_chart(df.groupby("Year")["Global_Sales"].sum())

st.write("Top 5 Publishers en Ventas Globales")
top_publishers = pd.read_sql("SELECT * FROM top_publishers", conn)
st.table(top_publishers)
