import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def send_email(subject, body, sender, recipients, password, attachment_path=None):
    """Send an email with optional attachment using Gmail's SMTP server."""

    # Create email container
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    # Attach the email body
    msg.attach(MIMEText(body, "plain"))  # Change "plain" to "html" for HTML email

    # Attach file (if provided)
    if attachment_path:
        try:
            with open(attachment_path, "rb") as attachment:
                file_part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
            file_part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            msg.attach(file_part)
            print(f"📎 Attached file: {os.path.basename(attachment_path)}")
        except Exception as e:
            print(f"❌ Error attaching file: {e}")

    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender, password)
            smtp_server.sendmail(sender, recipients, msg.as_string())
        print("✅ Email sent successfully!")
    except smtplib.SMTPException as e:
        print(f"❌ Error sending email: {e}")

# Example Usage:
