cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)
print(cols['age'][:10])