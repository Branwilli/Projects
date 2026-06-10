import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

HOST = "smtp.gmail.com"
PORT = 587
SENDER_EMAIL = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

@app.route('/api/send-welcome-email', methods=['POST'])
def send_email(): 
    try: 
        data = request.get_json()
        receiver_email = data.get('receiver_email')
        first_name = data.get('first_name', '')
        #last_name = data.get('last_name', '')

        if not receiver_email:
            return jsonify({'success': False, 'message': 'Receiver email is required'}), 400
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = f"This is a test run, {first_name}"
        
        body = """
        Hello, this is a test run to simulate how automated emails are sent.

        Please do not responsed to this email.
        """

        msg.attach(MIMEText(body, 'plain'))

        filename = "Customers.csv"
        attachment = open(filename, 'rb')

        attachment_package = MIMEBase('application', 'ocset-stream')
        attachment_package.set_payload((attachment).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename)
        msg.attach(attachment_package)
        text = msg.as_string()

        with smtplib.SMTP(HOST, PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, receiver_email, text)
    
        return jsonify({
            'success': True,
            'message': f"Welcome email sent to {receiver_email}"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to send email: {str(e)}'
        }), 500
    
if __name__ == '__main__':
    app.run(debug=True)
