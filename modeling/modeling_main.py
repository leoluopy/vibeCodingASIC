import sys
import json
import copy
from collections import defaultdict
from model_tracing.config import chip_specs
from model_tracing.factory import get_modeler


COMPOSITE_NAMES = {
    'Qwen2DecoderLayer', 'LlamaDecoderLayer',
    'Qwen2Attention', 'LlamaAttention', 'Qwen2SdpaAttention',
    'Qwen2MLP', 'LlamaMLP', 'DeepseekV2MLP',
    'DeepseekV2MLAAttention',
}


def process_trace(trace_path, output_path=None):
    with open(trace_path, 'r') as f:
        data = json.load(f)

    events = data.get('traceEvents', data)
    x_events = [ev for ev in events if ev.get('ph') == 'X']

    x_events.sort(key=lambda ev: ev['ts'])

    # ============================================================
    # Pass 1: Forward propagation — estimate leaf modules, drift
    # ============================================================
    cumulative_drift = 0.0
    for ev in x_events:
        ev['ts'] += cumulative_drift

        name = ev.get('name', '')
        args = ev.get('args', {})
        modeler = get_modeler(name, chip_specs)

        if modeler is not None and name not in COMPOSITE_NAMES:
            old_dur = ev['dur']
            ev['dur'] = round(modeler.estimate(name, args), 3)
            cumulative_drift += (ev['dur'] - old_dur)

    # ============================================================
    # Pass 2: Recalculate composite durations from children
    # ============================================================
    composites_by_name = defaultdict(list)
    for ev in x_events:
        if ev.get('name') in COMPOSITE_NAMES:
            composites_by_name[ev['name']].append(ev)

    # Composite names that are "outer" (contain other composites)
    OUTER_COMPOSITES = {'Qwen2DecoderLayer', 'LlamaDecoderLayer'}
    # Inner composites (attention, mlp) — in flat mode, keep modeler estimate

    # Process inner composites first, then outer
    composite_order = [
        'Qwen2Attention', 'LlamaAttention', 'Qwen2SdpaAttention', 'DeepseekV2MLAAttention',
        'Qwen2MLP', 'LlamaMLP', 'DeepseekV2MLP',
        'Qwen2DecoderLayer', 'LlamaDecoderLayer',
    ]

    for cname in composite_order:
        if cname not in composites_by_name:
            continue
        cevs = composites_by_name[cname]
        cevs.sort(key=lambda ev: ev['ts'])

        for k, cev in enumerate(cevs):
            cpath = cev['args'].get('module_path', cev['name'])
            children = []

            if '.' in cpath:
                # Nested module_path → prefix matching
                prefix = cpath + '.'
                for ev in x_events:
                    if ev is cev:
                        continue
                    epath = ev['args'].get('module_path', '')
                    if epath.startswith(prefix):
                        children.append(ev)
            elif cname in OUTER_COMPOSITES:
                # Flat outer composite → temporal bracketing
                cstart = cev['ts']
                cend = cevs[k + 1]['ts'] if k + 1 < len(cevs) else float('inf')
                for ev in x_events:
                    if ev is cev:
                        continue
                    if ev['ts'] > cstart and ev['ts'] < cend:
                        if ev.get('name') not in COMPOSITE_NAMES:
                            children.append(ev)
            # Inner composite in flat mode → skip, keep modeler estimate

            if children:
                max_child_end = max(ev['ts'] + ev['dur'] for ev in children)
                cev['dur'] = round(max_child_end - cev['ts'], 3)

    # ============================================================
    # Output
    # ============================================================
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        with open(trace_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Print summary
    results = []
    for ev in events:
        if ev.get('ph') != 'X':
            continue
        name = ev.get('name', '')
        dur = ev.get('dur', None)
        results.append((name, dur))

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
