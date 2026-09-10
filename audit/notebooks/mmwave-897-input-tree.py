import os


def tree(p, depth=0, maxdepth=3):
    if depth > maxdepth:
        return
    try:
        entries = sorted(os.listdir(p))
    except Exception as e:
        print('  ' * depth + f'{os.path.basename(p)}: ERR {e}')
        return
    print('  ' * depth + f'{os.path.basename(p)}/ ({len(entries)})')
    for e in entries[:20]:
        full = os.path.join(p, e)
        if os.path.isdir(full):
            tree(full, depth + 1, maxdepth)
        else:
            print('  ' * (depth + 1) + e)


tree('/kaggle/input')
