"""Execute selected assignment notebooks with the invoking Python environment."""
import argparse
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

class CurrentPythonKernel(KernelManager):
    def format_kernel_cmd(self, extra_arguments=None):
        return [sys.executable, '-m', 'ipykernel_launcher', '-f', self.connection_file] + (extra_arguments or [])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('orders', nargs='*', type=int, help='Notebook numbers 1–16; default: all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for order in args.orders or range(1, 17):
        if not 1 <= order <= 16:
            parser.error('Notebook numbers must be between 1 and 16')
        path = root / 'notebooks' / f'{order:02d}' / 'notebook.ipynb'
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=600, allow_errors=False,
                                kernel_manager_class=CurrentPythonKernel,
                                resources={'metadata': {'path': str(path.parent)}})
        client.execute()
        nbformat.write(notebook, path)
        print(f'{order:02d}: executed successfully, outputs saved')

if __name__ == '__main__':
    main()
