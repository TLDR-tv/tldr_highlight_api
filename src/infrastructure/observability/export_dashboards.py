#!/usr/bin/env python
"""Export Logfire dashboard configurations to JSON.

This script exports all dashboard configurations to a JSON file that can be
imported into Logfire or used for infrastructure as code.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.infrastructure.observability.dashboards import DashboardConfig


def main():
    """Export all dashboard configurations."""
    print("Exporting TL;DR Highlight API dashboards...")
    
    # Get all dashboard configurations
    config = DashboardConfig.export_all_dashboards()
    
    # Create output directory
    output_dir = Path(__file__).parent / "configs"
    output_dir.mkdir(exist_ok=True)
    
    # Write main configuration
    output_file = output_dir / "logfire_dashboards.json"
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Exported all dashboards to {output_file}")
    
    # Also export individual dashboards
    for dashboard in config["dashboards"]:
        dashboard_file = output_dir / f"{dashboard['name'].lower().replace(' ', '_')}.json"
        with open(dashboard_file, "w") as f:
            json.dump(dashboard, f, indent=2)
        print(f"✅ Exported {dashboard['name']} to {dashboard_file}")
    
    # Export alerts separately
    alerts_file = output_dir / "logfire_alerts.json"
    with open(alerts_file, "w") as f:
        json.dump({"alerts": config["alerts"]}, f, indent=2)
    print(f"✅ Exported alerts to {alerts_file}")
    
    print("\n📊 Dashboard Summary:")
    print(f"  - Total dashboards: {len(config['dashboards'])}")
    print(f"  - Total alerts: {len(config['alerts'])}")
    
    total_panels = sum(len(d.get("panels", [])) for d in config["dashboards"])
    print(f"  - Total panels: {total_panels}")
    
    print("\n📝 Usage:")
    print("  1. Import these configurations into Logfire UI")
    print("  2. Or use them with Logfire API to programmatically create dashboards")
    print("  3. Customize thresholds and notification channels as needed")


if __name__ == "__main__":
    main()