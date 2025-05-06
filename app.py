import os
import pickle
import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import traceback

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and vectorizer
model = load_model("model/spam_model.keras")
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
spam_logs = []
prediction_stats = defaultdict(int)
spam_over_time_counts = defaultdict(int)

def authenticate_user(email):
    creds = None
    token_path = f'tokens/{email}_token.json'
    os.makedirs('tokens', exist_ok=True)

    # Load existing token if available
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as e:
            print("Failed to load credentials from token. Removing corrupted token.")
            os.remove(token_path)
            creds = None

    # Refresh or obtain new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print("Token refresh failed (likely from deleted client). Deleting token.")
                if os.path.exists(token_path):
                    os.remove(token_path)
                creds = None

        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                print("Error during re-authentication flow.")
                traceback.print_exc()
                raise e

    return build('gmail', 'v1', credentials=creds)

def fetch_recent_emails(service, email_id):
    results = service.users().messages().list(userId=email_id, maxResults=10).execute()
    messages = results.get('messages', [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId=email_id, id=msg['id']).execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])

        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "")

        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    break
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')

        if body.strip():
            emails.append({
                "subject": subject,
                "sender": sender,
                "content": body
            })

    return emails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan_emails', methods=['GET'])
def scan_emails():
    email_id = request.args.get('email', 'me')
    try:
        print(f"[AUTH] Authenticating user for email: {email_id}")
        service = authenticate_user(email_id)
        print("[AUTH] Success.")

        emails = fetch_recent_emails(service, email_id)
        print(f"[FETCH] Retrieved {len(emails)} emails.")

        if not emails:
            return jsonify({"status": "No emails found."})

        results = []
        for email in emails:
            email_features = vectorizer.transform([email["content"]]).toarray()
            prediction_prob = model.predict(email_features)[0][0]
            prediction = "Spam" if prediction_prob > 0.4 else "Normal"

            prediction_stats[prediction] += 1
            today = datetime.now().strftime("%Y-%m-%d")
            spam_logs.append({"date": today, "prediction": prediction})
            if prediction == "Spam":
                spam_over_time_counts[today] += 1

            results.append({
                "subject": email["subject"],
                "sender": email["sender"],
                "prediction": prediction,
                "accuracy": f"{prediction_prob * 100:.2f}%"
            })

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    features = vectorizer.transform([email_text]).toarray()
    prediction_prob = model.predict(features)[0][0]
    prediction = "Spam" if prediction_prob > 0.4 else "Normal"

    prediction_stats[prediction] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    spam_logs.append({"date": today, "prediction": prediction})
    if prediction == "Spam":
        spam_over_time_counts[today] += 1

    return jsonify({"prediction": prediction, "accuracy": f"{prediction_prob * 100:.2f}%"})

@app.route('/spam_over_time')
def spam_over_time():
    if not spam_over_time_counts:
        return jsonify({"labels": ["No Data"], "data": [0]})

    sorted_data = sorted(spam_over_time_counts.items())[-10:]
    labels = [d[0] for d in sorted_data]
    data = [d[1] for d in sorted_data]
    return jsonify({"labels": labels, "data": data})

@app.route('/reports')
def reports():
    labels = ['Spam', 'Normal']
    values = [prediction_stats['Spam'], prediction_stats['Normal']]
    if sum(values) == 0:
        values = [1, 1]

    plt.figure(figsize=(5, 5))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['red', 'green'])
    plt.title('Spam Detection Statistics')
    plt.savefig('static/report.png')
    return render_template('reports.html', report_image='static/report.png')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
