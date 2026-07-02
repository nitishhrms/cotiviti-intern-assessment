"""Parse structured API logs and print a summary dashboard."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyse(log_file: str):
    path = Path(log_file)
    if not path.exists():
        print(f"Log file not found: {log_file}")
        sys.exit(1)

    total = 0
    errors = 0
    latencies = []
    report_lengths = []
    pathology_counts: Counter = Counter()
    hourly: Counter = Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            latencies.append(record.get("processing_time_ms", 0))
            if record.get("report_length"):
                report_lengths.append(record["report_length"])
            if record.get("top_pathology"):
                pathology_counts[record["top_pathology"]] += 1
            if record.get("error"):
                errors += 1
            if record.get("timestamp"):
                hour = str(record["timestamp"])[:13]
                hourly[hour] += 1

    if total == 0:
        print("No log entries found.")
        return

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    error_rate = errors / total * 100

    print("=" * 50)
    print("X-Ray API  —  Log Analysis")
    print("=" * 50)
    print(f"Total requests  : {total}")
    print(f"Errors          : {errors}  ({error_rate:.1f}%)")
    print(f"Avg latency     : {avg_latency:.0f} ms")
    if report_lengths:
        print(f"Avg report len  : {sum(report_lengths)/len(report_lengths):.0f} tokens")
    print()
    print("Top 3 pathologies:")
    for path_, count in pathology_counts.most_common(3):
        print(f"  {path_:<30} {count}")
    print()
    print("Requests per hour (last 10):")
    for hour in sorted(hourly)[-10:]:
        bar = "█" * min(hourly[hour], 40)
        print(f"  {hour}  {bar} {hourly[hour]}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "api.log"
    analyse(log_path)
