#!/usr/bin/env python3
"""
Quick script to add navigation to existing HTML reports.
"""

from pathlib import Path

# Navigation styles to add
NAV_STYLES = """        .nav-link {
            position: absolute;
            top: 20px;
            left: 20px;
            background: #2b3139;
            color: #f0b90b;
            padding: 10px 20px;
            border-radius: 4px;
            text-decoration: none;
            border: 1px solid #f0b90b;
            transition: all 0.3s;
            font-weight: 600;
        }
        .nav-link:hover {
            background: #f0b90b;
            color: #0b0e11;
        }
"""

# Navigation link HTML
NAV_LINK = '            <a href="index.html" class="nav-link">← Back to Index</a>\n'

def add_navigation_to_report(html_path: Path):
    """Add navigation to a report HTML file."""
    print(f"Processing: {html_path.name}")

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if navigation already exists
    if 'Back to Index' in content:
        print(f"  ✓ Navigation already exists, skipping")
        return

    # Add position: relative to header style
    if 'text-align: center;' in content and 'position: relative;' not in content:
        content = content.replace(
            'text-align: center;',
            'text-align: center;\n            position: relative;'
        )
        print(f"  ✓ Added position: relative to header")

    # Add navigation styles before closing </style>
    if NAV_STYLES.strip() not in content:
        content = content.replace('</style>', NAV_STYLES + '    </style>')
        print(f"  ✓ Added navigation styles")

    # Add navigation link after <div class="header">
    if '<div class="header">' in content:
        content = content.replace(
            '<div class="header">\n',
            '<div class="header">\n' + NAV_LINK
        )
        print(f"  ✓ Added navigation link")

    # Save updated content
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✅ Updated successfully\n")

def main():
    """Add navigation to all report HTML files."""
    data_dir = Path(__file__).parent.parent / "data" / "data_sources"

    # Reports to update (exclude index.html)
    reports = [
        "consolidation_report.html",
        "market_analysis_report.html",
        "evolutive_report.html"
    ]

    print("Adding navigation to reports...\n")

    for report in reports:
        html_path = data_dir / report
        if html_path.exists():
            add_navigation_to_report(html_path)
        else:
            print(f"❌ Not found: {report}\n")

    print("✨ Done!")

if __name__ == "__main__":
    main()
