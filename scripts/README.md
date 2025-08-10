# Scripts Directory

This directory contains setup and installation scripts for the ORDeC project.

## install-krakatau.sh

Installs Krakatau (krak2), a Java bytecode analysis toolkit. This script:

1. Clones the Krakatau repository from GitHub
2. Builds the Rust version using Cargo
3. Creates a symlink to the krak2 binary in `~/.local/bin`
4. Adds `~/.local/bin` to PATH if needed

### Requirements

- Rust and Cargo (installed via rustup.rs)
- Git

### Usage

```bash
bash scripts/install-krakatau.sh
```

After installation, you can use the `krak2` command for Java bytecode analysis:

```bash
krak2 dis --out temp MyClass.class  # Disassemble Java class file
krak2 asm --out MyClass.class temp/MyClass.j  # Reassemble from assembly
```

For more information about Krakatau, see: https://github.com/Storyyeller/Krakatau