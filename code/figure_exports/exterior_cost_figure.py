from pathlib import Path
import subprocess
import sys
if __name__ == '__main__':
    subprocess.run([sys.executable, str(Path(__file__).with_name('composite_figures.py')), '--only', 'exterior-cost', *sys.argv[1:]], check=True)
