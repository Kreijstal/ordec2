#!/bin/bash
# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

# Script to install Krakatau (krak2) - Java bytecode analysis toolkit

set -e

echo "Installing Krakatau (krak2)..."

# Create directory for Krakatau
KRAKATAU_DIR="$HOME/krakatau"
mkdir -p "$KRAKATAU_DIR"

cd "$KRAKATAU_DIR"

# Clone Krakatau repository
if [ ! -d "Krakatau" ]; then
    echo "Cloning Krakatau repository..."
    git clone https://github.com/Storyyeller/Krakatau.git
else
    echo "Krakatau repository already exists, updating..."
    cd Krakatau
    git pull
    cd ..
fi

cd Krakatau

# Make sure we have Rust and Cargo
echo "Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust and Cargo are required to build Krakatau"
    echo "Please install Rust first: https://rustup.rs/"
    exit 1
fi

cargo --version

# Build Krakatau with Cargo
echo "Building Krakatau..."
cargo build --release

# Create symlink to krak2 binary
echo "Creating krak2 symlink..."
ln -sf "$HOME/krakatau/Krakatau/target/release/krak2" "$HOME/.local/bin/krak2"

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Krakatau (krak2) installation completed!"
echo "You can now use 'krak2' command. You may need to reload your shell or run:"
echo "  source ~/.bashrc"