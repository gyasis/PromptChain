"""
Reporting utilities for UseCase Validator
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from jinja2 import Template
from src.utils.logger import get_logger

logger = get_logger(__name__)

class UseCaseReporter:
    """Generates comprehensive reports for use case test results."""

    def __init__(self):
        self.reports_dir = 'reports'
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_html_report(self, results: List[Dict[str, Any]], report_name: str = None) -> str:
        """Generate HTML report from test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name or 'usecase_report'}_{timestamp}.html"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # HTML template
        html_template = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UseCase Validator Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card.total { background-color: #e3f2fd; }
        .summary-card.passed { background-color: #e8f5e8; }
        .summary-card.failed { background-color: #ffebee; }
        .summary-card.rate { background-color: #fff3e0; }
        .test-results { margin-top: 30px; }
        .test-item { margin-bottom: 20px; padding: 15px; border-radius: 8px; border-left: 4px solid; }
        .test-item.passed { background-color: #f1f8e9; border-left-color: #4caf50; }
        .test-item.failed { background-color: #ffebee; border-left-color: #f44336; }
        .test-name { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
        .test-status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .test-status.passed { background-color: #4caf50; color: white; }
        .test-status.failed { background-color: #f44336; color: white; }
        .test-output { margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; font-family: monospace; font-size: 12px; white-space: pre-wrap; }
        .timestamp { color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>UseCase Validator Report</h1>
            <p class="timestamp">Generated on {{ timestamp }}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card total">
                <h3>Total Tests</h3>
                <h2>{{ total_tests }}</h2>
            </div>
            <div class="summary-card passed">
                <h3>Passed</h3>
                <h2>{{ passed_tests }}</h2>
            </div>
            <div class="summary-card failed">
                <h3>Failed</h3>
                <h2>{{ failed_tests }}</h2>
            </div>
            <div class="summary-card rate">
                <h3>Success Rate</h3>
                <h2>{{ "%.1f"|format(success_rate) }}%</h2>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results</h2>
            {% for result in results %}
            <div class="test-item {{ result.status.lower() }}">
                <div class="test-name">{{ result.usecase_name }}</div>
                <span class="test-status {{ result.status.lower() }}">{{ result.status }}</span>
                {% if result.output %}
                <div class="test-output">{{ result.output }}</div>
                {% endif %}
                {% if result.errors %}
                <div class="test-output" style="background-color: #ffebee;">{{ result.errors }}</div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        ''')
        
        html_content = html_template.render(
            results=results,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")
        return filepath

    def generate_json_report(self, results: List[Dict[str, Any]], report_name: str = None) -> str:
        """Generate JSON report from test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name or 'usecase_report'}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'PASSED'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate
            },
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report generated: {filepath}")
        return filepath

    def send_slack_notification(self, results: List[Dict[str, Any]]):
        """Send Slack notification with test results."""
        try:
            import requests
            
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                logger.warning("SLACK_WEBHOOK_URL not configured")
                return
            
            total_tests = len(results)
            passed_tests = len([r for r in results if r['status'] == 'PASSED'])
            failed_tests = total_tests - passed_tests
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            # Determine color based on success rate
            color = "good" if success_rate >= 80 else "warning" if success_rate >= 60 else "danger"
            
            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": "UseCase Validator Report",
                        "fields": [
                            {
                                "title": "Total Tests",
                                "value": str(total_tests),
                                "short": True
                            },
                            {
                                "title": "Passed",
                                "value": str(passed_tests),
                                "short": True
                            },
                            {
                                "title": "Failed",
                                "value": str(failed_tests),
                                "short": True
                            },
                            {
                                "title": "Success Rate",
                                "value": f"{success_rate:.1f}%",
                                "short": True
                            }
                        ],
                        "footer": "UseCase Validator",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def send_email_notification(self, results: List[Dict[str, Any]]):
        """Send email notification with test results."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Email configuration
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', 587))
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')
            email_to = os.getenv('EMAIL_TO')
            
            if not all([email_user, email_password, email_to]):
                logger.warning("Email configuration incomplete")
                return
            
            total_tests = len(results)
            passed_tests = len([r for r in results if r['status'] == 'PASSED'])
            failed_tests = total_tests - passed_tests
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            # Create email content
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = email_to
            msg['Subject'] = f"UseCase Validator Report - {success_rate:.1f}% Success Rate"
            
            body = f"""
UseCase Validator Test Results

Summary:
- Total Tests: {total_tests}
- Passed: {passed_tests}
- Failed: {failed_tests}
- Success Rate: {success_rate:.1f}%

Detailed Results:
"""
            
            for result in results:
                status_icon = "✅" if result['status'] == 'PASSED' else "❌"
                body += f"{status_icon} {result['usecase_name']}: {result['status']}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, email_to, text)
            server.quit()
            
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")




