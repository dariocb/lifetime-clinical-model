"""Generate synthetic tables using the canonical package implementation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifetime_clinical.data.synthetic import SyntheticConfig, generate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "synthetic_data.json",
                        help="JSON object containing SyntheticConfig settings")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "generated_synthetic",
                        help="Directory for generated CSV tables and metadata")
    args = parser.parse_args()
    try:
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("Configuration must be a JSON object")
        config = SyntheticConfig(**settings)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Cannot load configuration {args.config}: {exc}")
    dataset = generate_dataset(config)
    dataset.save(args.output)
    print(f"Saved {len(dataset.patients)} patients and {len(dataset.events)} events to {args.output.resolve()}")


if __name__ == "__main__":
    main()
