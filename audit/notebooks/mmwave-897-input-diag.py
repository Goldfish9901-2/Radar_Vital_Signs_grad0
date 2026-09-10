import os
import sys

print('CWD:', os.getcwd())
print('sys.path:', sys.path)
if os.path.exists('/kaggle/input'):
    print('/kaggle/input:', os.listdir('/kaggle/input'))
    for d in sorted(os.listdir('/kaggle/input')):
        p = f'/kaggle/input/{d}'
        try:
            print(f'  {d}: {os.listdir(p)[:8]}')
        except Exception as e:
            print(f'  {d}: ERR {e}')
else:
    print('/kaggle/input missing')
