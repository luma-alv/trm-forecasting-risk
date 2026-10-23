"""Daily forecast command-line pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml
from trm_forecast.data.ingestion import load_trm
from trm_forecast.models.baseline import forecast_last


def run(input_path, output_path, config_path="configs/base.yaml"):
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data = load_trm(input_path, config["data"]["date_column"], config["data"]["value_column"])
    horizon = int(config["forecast"]["horizon"])
    forecast = forecast_last(data[config["data"]["value_column"]], horizon)
    result = forecast.rename("forecast").to_frame()
    result["model"] = "naive_last"
    result.index.name = "date"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the daily TRM forecast")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    run(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
