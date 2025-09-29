import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import re

# Load data
data = pd.read_csv("human_ai_responses_fulll.csv", encoding="ISO-8859-1")

# Add Cleaned_Response column
def remove_symbols(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;]', '', str(text)) # Ensure text is string
    text = re.sub(r'\s+', ' ', text).strip()
    return text
data['Cleaned_Response'] = data['Response'].apply(remove_symbols)

# Sidebar filters
st.sidebar.title("Filters")
source = st.sidebar.selectbox("Choose Source", ["All", "Human", "AI"])
tone = st.sidebar.selectbox("Choose Tone", ["All"] + list(data["Tone"].unique()))

filtered = data.copy()
if source != "All":
    filtered = filtered[filtered["Source Type"] == source]
if tone != "All":
    filtered = filtered[filtered["Tone"] == tone]

st.title("💭 What does it mean to be human?")
st.write("An interactive comparison of **Human vs AI responses**")

# ---- Sentiment distribution ----
filtered["Sentiment"] = filtered["Response"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
st.subheader("📊 Sentiment Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.boxplot(x="Source Type", y="Sentiment", data=filtered, ax=ax)
    ax.set_title("Sentiment Scores (Boxplot)")
    st.pyplot(fig)

with col2:
    avg_sent = filtered.groupby("Source Type")["Sentiment"].mean()
    fig, ax = plt.subplots()
    avg_sent.plot(kind="bar", color=["#3498db", "#e74c3c"], ax=ax)
    ax.set_title("Average Sentiment by Source")
    ax.set_ylabel("Sentiment Polarity")
    st.pyplot(fig)

# ---- Word clouds side by side ----
st.subheader("☁️ Word Clouds")

human_text = " ".join(data[data["Source Type"] == "Human"]["Cleaned_Response"].astype(str))
ai_text = " ".join(data[data["Source Type"] == "AI"]["Cleaned_Response"].astype(str))

col1, col2 = st.columns(2)

with col1:
    st.markdown("**👤 Human Responses**")
    wc_human = WordCloud(width=600, height=400, background_color="white").generate(human_text)
    fig, ax = plt.subplots()
    ax.imshow(wc_human, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.markdown("**🤖 AI Responses**")
    wc_ai = WordCloud(width=600, height=400, background_color="white").generate(ai_text)
    fig, ax = plt.subplots()
    ax.imshow(wc_ai, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ---- Response Explorer ----
st.subheader("📝 Response Explorer")
for _, row in filtered.iterrows():
    st.markdown(f"**{row['Source Type']}** ({row['Tone']}): {row['Cleaned_Response']}")
