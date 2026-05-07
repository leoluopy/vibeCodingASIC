import sys
import json
import copy
from model_tracing.config import chip_specs
from model_tracing.factory import get_modeler


def process_trace(trace_path, output_path=None):
    with open(trace_path, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', data)
    results = []

    for ev in events:
        if ev.get('ph') != 'X':
            continue

        name = ev.get('name', '')
        args = ev.get('args', {})

        modeler = get_modeler(name, chip_specs)
        if modeler is not None:
            estimated_dur = modeler.estimate(name, args)
            estimated_dur = round(estimated_dur, 3)
            ev['dur'] = estimated_dur
            results.append((name, estimated_dur))
        else:
            results.append((name, None))

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open(trace_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"{'Module':45s} {'Estimated Dur (μs)':>20s}")
    print('-' * 67)
    for name, dur in results:
        if dur is not None:
            print(f"{name:45s} {dur:>20.3f}")
        else:
            print(f"{name:45s} {'N/A':>20s}")

    total = sum(d for _, d in results if d is not None)
    print('-' * 67)
    print(f"{'TOTAL':45s} {total:>20.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 modeling_main.py <trace.json> [output.json]")
        sys.exit(1)

    trace_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    process_trace(trace_path, output_path)


if __name__ == '__main__':
    main()
