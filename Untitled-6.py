for h, vals in cols.items():
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')