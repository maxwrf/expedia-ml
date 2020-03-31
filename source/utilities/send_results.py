import os
import smtplib
import ssl
import logging
import unicodedata
from email.mime.text import MIMEText


def get_credentials():
    email = os.getenv("EMAIL").strip()
    password = os.getenv("PASSWORD").strip()
    return email, password


def read_content():
    with open('logs/expedia.log', 'r') as f:
        logs = f.read()
    return logs.strip()


def send_email():
    logger = logging.getLogger('pipeline.email')
    sender, password = get_credentials()
    receiver = os.getenv('RECEIVER')
    message = read_content()
    message = unicodedata.normalize("NFKD", message)
    message = MIMEText(message)
    message['Subject'] = 'Expedia logs'
    message['To'] = receiver
    message['From'] = sender
    port = 465
    context = ssl.create_default_context()
    logger.info('Start sending logs by email')
    # print(sender, password, message)
    with smtplib.SMTP_SSL('smtp.gmail.com', port, context=context) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg=message.as_string())
