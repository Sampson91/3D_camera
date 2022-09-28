import smtplib
from email.mime.text import MIMEText


def send_email(error_message):

    mail_host = "smtp.163.com"
    mail_user = ""
    mail_pass = ""
    sender = "@163.com"
    receivers = ["shanwei.mu@pd-automotive.com"]

    content = error_message
    message = MIMEText(content, 'plain', 'utf-8')
    message["Subject"] = "Result of your training"
    message["From"] = sender
    message["To"] = receivers[0]

    try:
        smtp_obj = smtplib.SMTP()
        smtp_obj.connect(mail_host, 25)
        # smtp_obj = smtplib.SMTP_SSL(mail_host)
        smtp_obj.login(mail_user, mail_pass)
        smtp_obj.sendmail(sender, receivers, message.as_string())
        smtp_obj.quit()
        print("success")
    except smtplib.SMTPException as failed_error:
        print("error", failed_error)

