#!/usr/bin/env python3
"""
Security Scanning Agent - Find API keys and credentials in git-tracked files
Uses PromptChain to intelligently identify secrets and sensitive data
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime


class SecurityScanner:
    """Scan git-tracked files for API keys and sensitive data"""

    # Common API key patterns
    PATTERNS = {
        'openai_key': r'sk-[A-Za-z0-9]{20,}',
        'anthropic_key': r'sk-ant-[A-Za-z0-9\-]{95,}',
        'generic_api_key': r'(?i)(api[_-]?key|apikey|api[_-]?secret|token)\s*[=:]\s*["\']?([A-Za-z0-9\-_]{20,})["\']?',
        'aws_access_key': r'AKIA[0-9A-Z]{16}',
        'github_token': r'ghp_[A-Za-z0-9]{36,}',
        'private_key_header': r'-----BEGIN (RSA |EC )?PRIVATE KEY-----',
        'generic_secret': r'(?i)(secret|password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']',
    }

    # File extensions to skip (binary files)
    SKIP_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
        '.pdf', '.zip', '.tar', '.gz', '.7z',
        '.mp4', '.avi', '.mov', '.mkv',
        '.pkl', '.h5', '.hdf5', '.pth', '.ckpt',
        '.pyc', '.so', '.o', '.a',
        '.parquet', '.arrow',
    }

    # Directories to skip
    SKIP_DIRS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        'build', 'dist', '.eggs', '.tox', '.nox',
    }

    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path).resolve()
        self.findings: List[Dict] = []
        self.scanned_files = 0
        self.skipped_files = 0

    def get_tracked_files(self) -> List[str]:
        """Get all git-tracked files"""
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to get tracked files: {result.stderr}")

        return [f for f in result.stdout.strip().split('\n') if f]

    def should_skip_file(self, filepath: str) -> bool:
        """Check if file should be skipped"""
        path = Path(filepath)

        # Skip by extension
        if path.suffix.lower() in self.SKIP_EXTENSIONS:
            return True

        # Skip by directory
        for part in path.parts:
            if part in self.SKIP_DIRS:
                return True

        return False

    def scan_file(self, filepath: str) -> List[Dict]:
        """Scan a single file for secrets"""
        full_path = self.repo_path / filepath
        findings = []

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Scan for each pattern
            for pattern_name, pattern in self.PATTERNS.items():
                for match in re.finditer(pattern, content):
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1

                    # Get the matched text (truncate if very long)
                    matched_text = match.group(0)
                    if len(matched_text) > 100:
                        matched_text = matched_text[:100] + '...'

                    findings.append({
                        'file': filepath,
                        'line': line_num,
                        'pattern': pattern_name,
                        'match': matched_text,
                        'context': self._get_context(content, match.start(), match.end())
                    })

        except Exception as e:
            # Skip files that can't be read as text
            pass

        return findings

    def _get_context(self, content: str, start: int, end: int, context_chars: int = 50) -> str:
        """Get context around a match"""
        context_start = max(0, start - context_chars)
        context_end = min(len(content), end + context_chars)
        context = content[context_start:context_end]

        # Replace newlines for readability
        context = context.replace('\n', '\\n')

        return context

    def scan_repository(self) -> Dict:
        """Scan all tracked files in the repository"""
        print(f"🔍 Scanning repository: {self.repo_path}")
        print(f"📂 Getting list of tracked files...")

        tracked_files = self.get_tracked_files()
        total_files = len(tracked_files)

        print(f"📊 Found {total_files} tracked files")
        print(f"🔎 Scanning for secrets and API keys...\n")

        for i, filepath in enumerate(tracked_files, 1):
            if self.should_skip_file(filepath):
                self.skipped_files += 1
                continue

            file_findings = self.scan_file(filepath)
            self.findings.extend(file_findings)
            self.scanned_files += 1

            # Progress update every 100 files
            if i % 100 == 0:
                print(f"   Scanned {i}/{total_files} files ({len(self.findings)} findings so far)...")

        # Generate report
        report = self._generate_report()

        return report

    def _generate_report(self) -> Dict:
        """Generate comprehensive security report"""
        # Group findings by file
        files_with_secrets = {}
        for finding in self.findings:
            filepath = finding['file']
            if filepath not in files_with_secrets:
                files_with_secrets[filepath] = []
            files_with_secrets[filepath].append(finding)

        # Group by pattern type
        by_pattern = {}
        for finding in self.findings:
            pattern = finding['pattern']
            if pattern not in by_pattern:
                by_pattern[pattern] = []
            by_pattern[pattern].append(finding)

        report = {
            'scan_time': datetime.now().isoformat(),
            'repository': str(self.repo_path),
            'statistics': {
                'total_tracked_files': self.scanned_files + self.skipped_files,
                'scanned_files': self.scanned_files,
                'skipped_files': self.skipped_files,
                'total_findings': len(self.findings),
                'files_with_secrets': len(files_with_secrets)
            },
            'findings_by_file': files_with_secrets,
            'findings_by_pattern': {k: len(v) for k, v in by_pattern.items()},
            'all_findings': self.findings
        }

        return report

    def print_report(self, report: Dict):
        """Print human-readable report"""
        print("\n" + "="*80)
        print("🔐 SECURITY SCAN REPORT")
        print("="*80)

        stats = report['statistics']
        print(f"\n📊 STATISTICS:")
        print(f"   Total tracked files: {stats['total_tracked_files']}")
        print(f"   Scanned files: {stats['scanned_files']}")
        print(f"   Skipped files: {stats['skipped_files']}")
        print(f"   Total findings: {stats['total_findings']}")
        print(f"   Files with secrets: {stats['files_with_secrets']}")

        if report['findings_by_pattern']:
            print(f"\n🔍 FINDINGS BY TYPE:")
            for pattern, count in sorted(report['findings_by_pattern'].items(),
                                        key=lambda x: x[1], reverse=True):
                print(f"   {pattern}: {count}")

        if report['findings_by_file']:
            print(f"\n⚠️  FILES CONTAINING SECRETS:")
            for filepath, findings in sorted(report['findings_by_file'].items()):
                print(f"\n   📄 {filepath} ({len(findings)} findings)")
                for finding in findings[:3]:  # Show first 3 per file
                    print(f"      Line {finding['line']}: {finding['pattern']}")
                    print(f"         Match: {finding['match'][:80]}...")
                if len(findings) > 3:
                    print(f"      ... and {len(findings) - 3} more")

        print("\n" + "="*80)

    def save_report(self, report: Dict, output_path: str):
        """Save detailed report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Detailed report saved to: {output_path}")

    def generate_purge_list(self, report: Dict, output_path: str):
        """Generate list of files to purge from git history"""
        files_to_purge = sorted(report['findings_by_file'].keys())

        with open(output_path, 'w') as f:
            for filepath in files_to_purge:
                f.write(f"{filepath}\n")

        print(f"📝 Purge list saved to: {output_path}")
        print(f"   Files to purge: {len(files_to_purge)}")

        return files_to_purge


def main():
    """Run security scan"""
    scanner = SecurityScanner()

    # Scan repository
    report = scanner.scan_repository()

    # Print report
    scanner.print_report(report)

    # Save detailed report
    report_path = '/tmp/security_scan_report.json'
    scanner.save_report(report, report_path)

    # Generate purge list
    purge_list_path = '/tmp/files-with-secrets.txt'
    files_to_purge = scanner.generate_purge_list(report, purge_list_path)

    # Summary
    print(f"\n✅ SCAN COMPLETE")
    print(f"   Found {report['statistics']['total_findings']} secrets in {report['statistics']['files_with_secrets']} files")

    if files_to_purge:
        print(f"\n⚠️  ACTION REQUIRED:")
        print(f"   Run git-filter-repo to remove these files from history:")
        print(f"   git filter-repo --invert-paths --paths-from-file {purge_list_path}")
    else:
        print(f"\n✅ No secrets found in tracked files!")


if __name__ == '__main__':
    main()
