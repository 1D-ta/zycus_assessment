import json
import os
import sys
from pathlib import Path
import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Zycus AI Support Suite",
    page_icon="🤖",
    layout="wide"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACCOUNTS_PATH = PROJECT_ROOT / "data" / "accounts.json"

st.title("🤖 Zycus AI Support Suite Dashboard")
st.markdown("---")

# Sidebar for Server Connection Check
st.sidebar.title("Configuration")
api_url = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000")

# Check Health
try:
    health_resp = requests.get(f"{api_url}/health", timeout=3.0)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success(" Connected to API Server")
        st.sidebar.metric("Loaded Tickets", health_data.get("tickets", 0))
        st.sidebar.metric("Loaded Accounts", health_data.get("accounts", 0))
        st.sidebar.metric("Loaded KB Chunks", health_data.get("kb_chunks", 0))
    else:
        st.sidebar.warning(f"⚠️ Connected, but server returned HTTP {health_resp.status_code}")
except Exception:
    st.sidebar.error("❌ Not connected to FastAPI server. Please start the server by running:\n`python -m app.main`")

# Create tabs for the two main tasks
tab1, tab2 = st.tabs(["🎫 Support Ticket Triage", "📈 TAM Account Health Summarizer"])

# ==========================================
# Tab 1: Ticket Triage
# ==========================================
with tab1:
    st.header("🎫 Intelligent Support Ticket Triage")
    st.markdown("Enter a support ticket's details to run sanitization, retrieve relevant knowledge-base articles, and classify fields.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Support Ticket Input")
        subject = st.text_input("Ticket Subject", value="Unable to sync CloudSync files - connection timed out")
        body = st.text_area(
            "Ticket Body",
            value="Hi team, I am trying to run the CloudSync service to upload our monthly audit reports. However, it fails with a socket connection error after 30 seconds. This is critical for our finance team. Please contact billing ops if needed at billingsupport@initech.com.",
            height=200
        )
        stream_option = st.checkbox("Stream response token-by-token", value=True)
        triage_button = st.button("Run Triage Pipeline", type="primary")

    with col2:
        st.subheader("Triage Pipeline Output")
        
        if triage_button:
            if not subject.strip() or not body.strip():
                st.error("Please fill in both the Subject and Body.")
            else:
                st.info("Sanitizing PII and executing retrieval...")
                
                if stream_option:
                    # Stream logic using Server-Sent Events (SSE)
                    st.write("Streaming Triage Response:")
                    placeholder = st.empty()
                    accumulated_text = ""
                    
                    try:
                        with requests.post(
                            f"{api_url}/triage/stream",
                            json={"subject": subject, "body": body},
                            stream=True,
                            timeout=60.0
                        ) as r:
                            r.raise_for_status()
                            for line in r.iter_lines():
                                if line:
                                    decoded_line = line.decode('utf-8')
                                    if decoded_line.startswith("data:"):
                                        token = decoded_line[5:].strip()
                                        if token == "[DONE]":
                                            break
                                        accumulated_text += token
                                        placeholder.code(accumulated_text, language="json")
                    except Exception as e:
                        st.error(f"Error during streaming triage: {e}")
                else:
                    # Non-stream block
                    try:
                        with st.spinner("Classifying ticket fields..."):
                            resp = requests.post(
                                f"{api_url}/triage",
                                json={"subject": subject, "body": body},
                                timeout=60.0
                            )
                            if resp.status_code == 200:
                                st.success("Triage classification complete!")
                                st.json(resp.json())
                            else:
                                st.error(f"Error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}")

# ==========================================
# Tab 2: TAM Summarizer
# ==========================================
with tab2:
    st.header("📈 Technical Account Management (TAM) Summarizer")
    st.markdown("Select a client account to query support tickets over the last 90 days and compile an executive brief with quote verification.")

    # Load account metadata from file to populate selectbox
    accounts = []
    if ACCOUNTS_PATH.is_file():
        try:
            with open(ACCOUNTS_PATH, "r", encoding="utf-8") as f:
                accounts = json.load(f)
        except Exception as e:
            st.error(f"Could not load accounts list: {e}")
    
    if accounts:
        account_options = {f"{acc['account_id']} - {acc.get('company', 'Unknown')}": acc["account_id"] for acc in accounts}
        selected_label = st.selectbox("Select Client Account", options=list(account_options.keys()))
        account_id = account_options[selected_label]
        
        # Display selected account profile info
        selected_acc = next((a for a in accounts if a["account_id"] == account_id), None)
        if selected_acc:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("TAM Owner", selected_acc.get("tam", "N/A"))
            col2.metric("Contract Value (ARR)", f"${selected_acc.get('arr_usd', 0):,}")
            col3.metric("Account Health", selected_acc.get("health_status", "N/A"))
            col4.metric("Seats Licensed", f"{selected_acc.get('seats_active', 0)} / {selected_acc.get('seats_licensed', 0)}")

            st.write("**Active Products:**", ", ".join(selected_acc.get("products", [])))
            if selected_acc.get("escalation_notes"):
                st.warning("⚠️ **Escalation Notes:** " + " | ".join(selected_acc["escalation_notes"]))
        
        stream_tam = st.checkbox("Stream narrative brief synthesis", value=True)
        generate_tam_btn = st.button("Generate Account Brief", type="primary")
        
        if generate_tam_btn:
            st.info("Filtering 90-day ticket window & extracting risks...")
            
            if stream_tam:
                st.write("Streaming Brief Synthesis:")
                placeholder = st.empty()
                accumulated_text = ""
                
                try:
                    with requests.get(
                        f"{api_url}/account/{account_id}/brief/stream",
                        stream=True,
                        timeout=90.0
                    ) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith("data:"):
                                    token = decoded_line[5:].strip()
                                    if token == "[DONE]":
                                        break
                                    accumulated_text += token
                                    placeholder.code(accumulated_text, language="json")
                except Exception as e:
                    st.error(f"Error during streaming brief: {e}")
            else:
                try:
                    with st.spinner("Running Stage 2 narrative synthesis..."):
                        resp = requests.get(f"{api_url}/account/{account_id}/brief", timeout=90.0)
                        if resp.status_code == 200:
                            st.success("Executive brief successfully generated!")
                            st.json(resp.json())
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
    else:
        st.warning("No accounts found in data/accounts.json.")

