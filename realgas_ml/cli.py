import argparse
import json
from .model import GasModel, gases

def main():
    parser = argparse.ArgumentParser(description="Real-gas ML properties in SI units")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--gas", choices=gases())
    parser.add_argument("--T", type=float, help="Temperature in kelvin")
    parser.add_argument("--P", type=float, help="Pressure in pascals")
    parser.add_argument("--n", type=float, default=1, help="Amount in mol")
    args = parser.parse_args()
    if args.list:
        print("\n".join(gases()))
        return
    if args.gas is None or args.T is None or args.P is None:
        parser.error("--gas, --T and --P are required unless --list is used")
    try:
        print(json.dumps(GasModel(args.gas).predict(args.T, args.P, args.n), indent=2))
    except ValueError as exc:
        parser.error(str(exc))

if __name__ == "__main__":
    main()

