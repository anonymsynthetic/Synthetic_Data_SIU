import io
import sys
import numpy as np
from scipy.stats import entropy

class StreamlitConsole:
    def __init__(self):
        self.buffer = io.StringIO()

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self.buffer
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout

    def get_console_output(self):
        return self.buffer.getvalue()

def calculate_kl_divergence(original_data, synthetic_data):
    min_val = min(original_data.min(), synthetic_data.min())
    max_val = max(original_data.max(), synthetic_data.max())
    bins = np.linspace(min_val, max_val, 50)
    hist_org, _ = np.histogram(original_data, bins=bins, density=True)
    hist_syn, _ = np.histogram(synthetic_data, bins=bins, density=True)
    hist_org = np.where(hist_org == 0, 1e-10, hist_org)
    hist_syn = np.where(hist_syn == 0, 1e-10, hist_syn)
    return entropy(hist_org, hist_syn)

def extract_last_epoch_losses(console_output):
    lines = console_output.split('\n')
    last_epoch_loss = next((line for line in reversed(lines) if 'Epoch:' in line and 'critic_loss:' in line and 'generator_loss:' in line), None)
    return last_epoch_loss

def format_epoch_loss(loss_str):
    import re
    if not loss_str:
        return loss_str
    # Remove 'Epoch: ... | ' part
    loss_str = re.sub(r'Epoch: \d+ \| ', '', loss_str)
    # Format numbers to 3 decimals
    loss_str = re.sub(r'(critic_loss:\s*)(-?\d+\.\d+)', lambda m: f"{m.group(1)}{float(m.group(2)):.3f}", loss_str)
    loss_str = re.sub(r'(generator_loss:\s*)(-?\d+\.\d+)', lambda m: f"{m.group(1)}{float(m.group(2)):.3f}", loss_str)
    return loss_str
