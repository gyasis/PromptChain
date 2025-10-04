"""
Command line interface for UseCase Validator
"""
import click
import os
import json
from pathlib import Path
from src.usecase_validator import UseCaseValidator
from src.utils.reporter import UseCaseReporter

@click.group()
def cli():
    """UseCase Validator - Convert plain English use cases to executable tests"""
    pass

@cli.command()
@click.argument('usecase_path')
@click.option('--tags', help='Run only use cases with specific tags')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--report', '-r', default='html', help='Report format: html, json, both, none')
@click.option('--report-name', help='Custom report name')
def run(usecase_path, tags, verbose, report, report_name):
    """Run use case tests from YAML specifications"""
    validator = UseCaseValidator()
    reporter = UseCaseReporter()
    
    if os.path.isfile(usecase_path):
        # Single use case file
        result = validator.execute_usecase(usecase_path)
        results = [result]  # Convert single result to list for reporting
        
        click.echo(f"UseCase: {result['usecase_name']}")
        click.echo(f"Status: {result['status']}")
        if verbose or result['status'] == 'FAILED':
            click.echo(f"Output:\n{result['output']}")
        if result['errors']:
            click.echo(f"Errors:\n{result['errors']}")
            
    elif os.path.isdir(usecase_path):
        # Directory of use case files
        yaml_files = list(Path(usecase_path).glob("*.yaml")) + list(Path(usecase_path).glob("*.yml"))
        
        results = []
        for yaml_file in yaml_files:
            if tags:
                # Filter by tags (simplified implementation)
                with open(yaml_file) as f:
                    content = f.read()
                if not any(tag in content for tag in tags.split(',')):
                    continue
            
            result = validator.execute_usecase(str(yaml_file))
            results.append(result)
            
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            click.echo(f"{status_icon} {result['usecase_name']}: {result['status']}")
        
        # Summary
        passed = len([r for r in results if r['status'] == 'PASSED'])
        total = len(results)
        click.echo(f"\nSummary: {passed}/{total} use cases passed")
        
        # AUTOMATIC REPORT GENERATION
        if report != 'none' and results:
            click.echo("\n📊 Generating reports...")
            
            if report in ['html', 'both']:
                html_file = reporter.generate_html_report(results, report_name)
                click.echo(f"📄 HTML report: {html_file}")
            
            if report in ['json', 'both']:
                json_file = reporter.generate_json_report(results, report_name)
                click.echo(f"📋 JSON report: {json_file}")
            
            # Send notifications if configured
            if os.getenv('SLACK_WEBHOOK_URL'):
                reporter.send_slack_notification(results)
                click.echo("📱 Slack notification sent")
            
            if os.getenv('EMAIL_NOTIFICATIONS_ENABLED', 'false').lower() == 'true':
                reporter.send_email_notification(results)
                click.echo("📧 Email notification sent")

@cli.command()
@click.argument('name')
def create(name):
    """Create a new use case template"""
    template = '''name: "{name}"
description: "Description of what this use case validates"
tags: ["e2e"]
timeout: 120

setup:
  - "Prepare test data and environment"

steps:
  - step: "First test step"
    action: "POST to /api/endpoint with data"
    expect: "Status 200 and expected response"

validations:
  - "Verify expected outcome"

cleanup:
  - "Clean up test data"
'''
    
    filename = f"usecases/{name.lower().replace(' ', '_')}.yaml"
    os.makedirs("usecases", exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(template.format(name=name))
    
    click.echo(f"Created use case template: {filename}")

@cli.command()
@click.argument('report_path')
@click.option('--format', 'report_format', default='html', help='Report format: html or json')
def report(report_path, report_format):
    """Generate report from existing JSON results"""
    if not os.path.exists(report_path):
        click.echo(f"Report file not found: {report_path}")
        return
    
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        reporter = UseCaseReporter()
        
        if report_format == 'html':
            report_file = reporter.generate_html_report(results)
            click.echo(f"HTML report generated: {report_file}")
        elif report_format == 'json':
            report_file = reporter.generate_json_report(results)
            click.echo(f"JSON report generated: {report_file}")
        else:
            click.echo("Invalid format. Use 'html' or 'json'")
            
    except Exception as e:
        click.echo(f"Error generating report: {e}")

@cli.command()
def setup():
    """Setup the UseCase Validator environment"""
    click.echo("🚀 Setting up UseCase Validator...")
    
    # Create necessary directories
    directories = ['usecases', 'reports', 'generated_tests', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        click.echo(f"✅ Created directory: {directory}")
    
    # Check .env file
    if not os.path.exists('.env'):
        click.echo("⚠️  .env file not found. Copy env.example to .env and configure your settings.")
    else:
        click.echo("✅ .env file exists")
    
    # Test database connection
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        if db.connection:
            click.echo("✅ Database connection successful")
        else:
            click.echo("⚠️  Database connection failed - check your DB_* environment variables")
    except Exception as e:
        click.echo(f"⚠️  Database setup issue: {e}")
    
    click.echo("\n🎉 Setup complete! You can now:")
    click.echo("1. Create a use case: python src/cli.py create 'My Test Case'")
    click.echo("2. Run use cases: python src/cli.py run usecases/")
    click.echo("3. View reports in the ./reports directory")

if __name__ == '__main__':
    cli()




