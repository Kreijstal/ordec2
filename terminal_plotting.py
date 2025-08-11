#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ORDeC contributors  
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced terminal plotting capabilities for ORDeC.

This module provides various methods to display waveforms in the terminal:
1. ASCII art plotting (always available)
2. Sixel graphics (if terminal supports it)
3. X11 plotting (if X server available)
"""

import os
import subprocess
import sys
import math
import termios
import tty
import select

def query_terminal_device_attributes():
    """
    Query terminal for Secondary Device Attributes (DA) to detect SIXEL support.
    
    Returns:
        str or None: Terminal response or None if no response/error
    """
    if not sys.stdout.isatty() or not sys.stdin.isatty():
        return None
    
    try:
        # Save terminal settings
        old_termios = termios.tcgetattr(sys.stdin.fileno())
        
        # Set terminal to raw mode with timeout
        tty.setraw(sys.stdin.fileno())
        new_termios = termios.tcgetattr(sys.stdin.fileno())
        new_termios[6][termios.VMIN] = 0  # Non-blocking read  
        new_termios[6][termios.VTIME] = 1  # 100ms timeout
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new_termios)
        
        # Send Secondary Device Attributes query
        sys.stdout.write('\033[>c')
        sys.stdout.flush()
        
        # Wait for response with timeout
        response = ""
        if select.select([sys.stdin], [], [], 0.1)[0]:  # 100ms timeout
            response = sys.stdin.read(32)  # Read up to 32 characters
        
        # Restore terminal settings
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_termios)
        
        return response if response else None
        
    except (OSError, termios.error, ImportError):
        # Restore terminal settings on error
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_termios)
        except:
            pass
        return None

def detect_sixel_support():
    """
    Detect SIXEL support using proper Secondary Device Attributes query.
    
    Returns:
        bool: True if SIXEL is supported, False otherwise
    """
    # Query terminal for device attributes
    response = query_terminal_device_attributes()
    
    if not response:
        # Fallback to basic terminal name detection
        term = os.environ.get('TERM', '').lower()
        # Known SIXEL-supporting terminals
        sixel_terms = ['xterm', 'mintty', 'mlterm', 'foot', 'wezterm', 'ghostty']
        return any(term_name in term for term_name in sixel_terms)
    
    # Parse the response: format is \e[>P1;P2;P3c
    # P1 indicates terminal type - values 2, 4, 64, 65 indicate SIXEL support
    import re
    match = re.match(r'\033\[>(\d+);.*?c', response)
    if match:
        terminal_type = int(match.group(1))
        # Terminal types that support SIXEL:
        # 2: VT240, VT241
        # 4: VT340, VT330 (and modern emulators like XTerm)
        # 64: VT520
        # 65: VT525
        return terminal_type in [2, 3, 4, 64, 65]
    
    return False

def detect_terminal_capabilities():
    """
    Comprehensive terminal capability detection using proper methods.
    
    Returns:
        dict: Dictionary with keys 'sixel', 'x11', 'ascii' and boolean values
    """
    capabilities = {
        'sixel': False,
        'x11': False,
        'ascii': True,  # Always available fallback
        'color': False,
        'unicode': False
    }
    
    # Check for color support
    try:
        colors = subprocess.run(['tput', 'colors'], capture_output=True, text=True, timeout=2)
        if colors.returncode == 0:
            color_count = int(colors.stdout.strip())
            capabilities['color'] = color_count >= 8
    except:
        # Fallback color detection
        term = os.environ.get('TERM', '')
        capabilities['color'] = any(x in term for x in ['256color', 'color', 'xterm'])
    
    # Check for Unicode support
    try:
        # Test if we can encode Unicode box drawing characters
        '┌─┐\n│ │\n└─┘'.encode(sys.stdout.encoding or 'utf-8')
        capabilities['unicode'] = True
    except:
        capabilities['unicode'] = False
    
    # Check for sixel support using proper DA query
    capabilities['sixel'] = detect_sixel_support()
    
    # Check for X11 server
    display = os.environ.get('DISPLAY')
    if display:
        try:
            result = subprocess.run(['xdpyinfo'], capture_output=True, timeout=2)
            capabilities['x11'] = result.returncode == 0
        except:
            # Try alternative method
            try:
                result = subprocess.run(['xwininfo', '-root'], capture_output=True, timeout=2)
                capabilities['x11'] = result.returncode == 0
            except:
                capabilities['x11'] = False
    
    return capabilities

def plot_ascii_basic(time_data, voltage_data, width=80, height=20, title="Waveform"):
    """Basic ASCII art plot using only standard ASCII characters."""
    if not time_data or not voltage_data:
        return "No data to plot"
    
    # Normalize data
    min_v, max_v = min(voltage_data), max(voltage_data)
    v_range = max_v - min_v if max_v != min_v else 1
    
    min_t, max_t = min(time_data), max(time_data)
    t_range = max_t - min_t if max_t != min_t else 1
    
    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot data points
    for t, v in zip(time_data, voltage_data):
        x = int((t - min_t) / t_range * (width - 1))
        y = height - 1 - int((v - min_v) / v_range * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = '*'
    
    # Add axes
    for y in range(height):
        grid[y][0] = '|'
    for x in range(width):
        grid[height-1][x] = '-'
    grid[height-1][0] = '+'
    
    # Convert to string
    lines = [title.center(width)]
    lines.extend(''.join(row) for row in grid)
    lines.append(f"Time: {min_t:.2e} to {max_t:.2e} s")
    lines.append(f"Voltage: {min_v:.3f} to {max_v:.3f} V")
    
    return '\n'.join(lines)

def plot_ascii_unicode(time_data, voltage_data, width=80, height=20, title="Waveform"):
    """Enhanced ASCII plot using Unicode box drawing characters."""
    if not time_data or not voltage_data:
        return "No data to plot"
    
    # Normalize data
    min_v, max_v = min(voltage_data), max(voltage_data)
    v_range = max_v - min_v if max_v != min_v else 1
    
    min_t, max_t = min(time_data), max(time_data)
    t_range = max_t - min_t if max_t != min_t else 1
    
    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot data points with different characters for better visibility
    plot_chars = ['●', '○', '▪', '▫', '♦', '◆']
    
    for i, (t, v) in enumerate(zip(time_data, voltage_data)):
        x = int((t - min_t) / t_range * (width - 1))
        y = height - 1 - int((v - min_v) / v_range * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            # Use different characters based on position for better visual effect
            grid[y][x] = plot_chars[i % len(plot_chars)]
    
    # Add Unicode box drawing axes
    for y in range(height):
        grid[y][0] = '┃' if y != height-1 else '┗'
    for x in range(1, width):
        grid[height-1][x] = '━'
    
    # Convert to string
    lines = [title.center(width)]
    lines.extend(''.join(row) for row in grid)
    lines.append(f"📊 Time: {min_t:.2e} to {max_t:.2e} s")
    lines.append(f"⚡ Voltage: {min_v:.3f} to {max_v:.3f} V")
    
    return '\n'.join(lines)

def plot_ascii_color(time_data, voltage_data, width=80, height=20, title="Waveform"):
    """Color ASCII plot using ANSI escape codes."""
    if not time_data or not voltage_data:
        return "No data to plot"
    
    # ANSI color codes
    RESET = '\033[0m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    
    # Normalize data
    min_v, max_v = min(voltage_data), max(voltage_data)
    v_range = max_v - min_v if max_v != min_v else 1
    
    min_t, max_t = min(time_data), max(time_data)
    t_range = max_t - min_t if max_t != min_t else 1
    
    # Create plot grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    colors = [[RESET for _ in range(width)] for _ in range(height)]
    
    # Plot data points with colors based on voltage level
    for t, v in zip(time_data, voltage_data):
        x = int((t - min_t) / t_range * (width - 1))
        y = height - 1 - int((v - min_v) / v_range * (height - 1))
        if 0 <= x < width and 0 <= y < height:
            # Color based on voltage level
            normalized_v = (v - min_v) / v_range
            if normalized_v > 0.8:
                color = BRIGHT_RED
            elif normalized_v > 0.6:
                color = RED
            elif normalized_v > 0.4:
                color = YELLOW
            elif normalized_v > 0.2:
                color = GREEN
            else:
                color = BLUE
                
            grid[y][x] = '●'
            colors[y][x] = color
    
    # Add colored axes
    for y in range(height):
        grid[y][0] = '│'
        colors[y][0] = CYAN
    for x in range(width):
        grid[height-1][x] = '─'
        colors[height-1][x] = CYAN
    grid[height-1][0] = '┼'
    colors[height-1][0] = CYAN
    
    # Convert to string with colors
    lines = [f"{MAGENTA}{title.center(width)}{RESET}"]
    for y in range(height):
        line = ""
        for x in range(width):
            line += f"{colors[y][x]}{grid[y][x]}{RESET}"
        lines.append(line)
    
    lines.append(f"{GREEN}Time: {min_t:.2e} to {max_t:.2e} s{RESET}")
    lines.append(f"{BRIGHT_GREEN}Voltage: {min_v:.3f} to {max_v:.3f} V{RESET}")
    
    return '\n'.join(lines)

def plot_sixel_placeholder(time_data, voltage_data, width=80, height=20, title="Waveform"):
    """Placeholder for sixel graphics (not implemented yet)."""
    return f"""
{title.center(width)}

[Sixel graphics would be displayed here]
- High resolution waveform plot
- Smooth curves with anti-aliasing
- Multiple colors for different signals

For now, falling back to ASCII plot:

{plot_ascii_basic(time_data, voltage_data, width, height, title)}
"""

def plot_waveform(time_data, voltage_data, width=80, height=20, title="Waveform", 
                 method='auto', capabilities=None):
    """
    Plot waveform using the best available method.
    
    Args:
        time_data: List of time points
        voltage_data: List of voltage values
        width: Plot width in characters
        height: Plot height in characters
        title: Plot title
        method: 'auto', 'ascii', 'unicode', 'color', 'sixel'
        capabilities: Terminal capabilities dict (auto-detected if None)
    """
    if capabilities is None:
        capabilities = detect_terminal_capabilities()
    
    if method == 'auto':
        if capabilities['sixel']:
            method = 'sixel'
        elif capabilities['color']:
            method = 'color' 
        elif capabilities['unicode']:
            method = 'unicode'
        else:
            method = 'ascii'
    
    if method == 'sixel':
        return plot_sixel_placeholder(time_data, voltage_data, width, height, title)
    elif method == 'color':
        return plot_ascii_color(time_data, voltage_data, width, height, title)
    elif method == 'unicode':
        return plot_ascii_unicode(time_data, voltage_data, width, height, title)
    else:
        return plot_ascii_basic(time_data, voltage_data, width, height, title)

def demo_plotting():
    """Demonstrate different plotting methods."""
    # Generate sample sine wave
    freq = 1000  # 1 kHz
    periods = 2
    points = 100
    
    time_data = [i * periods / (freq * points) for i in range(points)]
    voltage_data = [math.sin(2 * math.pi * freq * t) for t in time_data]
    
    print("=== Terminal Plotting Capabilities Demo ===\n")
    
    # Detect capabilities
    caps = detect_terminal_capabilities()
    print("Detected terminal capabilities:")
    for cap, available in caps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {cap.upper()}: {available}")
    print()
    
    # Show different plotting methods
    methods = ['ascii', 'unicode', 'color']
    
    for method in methods:
        if method == 'unicode' and not caps['unicode']:
            continue
        if method == 'color' and not caps['color']:
            continue
            
        print(f"=== {method.upper()} Plot ===")
        plot = plot_waveform(time_data, voltage_data, width=70, height=12,
                           title=f"1kHz Sine Wave ({method.title()})",
                           method=method, capabilities=caps)
        print(plot)
        print()

if __name__ == "__main__":
    demo_plotting()