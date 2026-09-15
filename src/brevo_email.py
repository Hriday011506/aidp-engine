import requests


def send_otp_email(api_key, sender_email, sender_name, recipient_email, otp):
    if not api_key or not sender_email or not recipient_email:
        return False, "Brevo email configuration is incomplete."

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json",
    }
    payload = {
        "sender": {"name": sender_name or "OptiRetail AI", "email": sender_email},
        "to": [{"email": recipient_email}],
        "subject": "Your OptiRetail AI verification code",
        "textContent": (
            f"Your OptiRetail AI verification code is {otp}. "
            "It expires in 5 minutes. If you did not request this code, ignore this email."
        ),
        "htmlContent": (
            "<div style='font-family:Arial,sans-serif;max-width:560px;margin:auto;'>"
            "<h2 style='color:#047857;'>OptiRetail AI</h2>"
            "<p>Your email verification code is:</p>"
            f"<div style='font-size:32px;font-weight:700;letter-spacing:8px;color:#047857;"
            "padding:18px 0;">{otp}</div>"
            "<p>This code expires in <strong>5 minutes</strong>.</p>"
            "</div>"
        ),
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if 200 <= response.status_code < 300:
            return True, "OTP sent successfully."
        return False, f"Brevo request failed ({response.status_code})."
    except requests.RequestException:
        return False, "Could not reach Brevo. Please try again."
