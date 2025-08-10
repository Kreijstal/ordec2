# ORDeC Development Container

This directory contains the devcontainer configuration for developing ORDeC in GitHub Codespaces or any devcontainer-compatible environment.

## What's Included

The devcontainer provides a complete development environment equivalent to the Docker setup, including:

### Core Tools
- **Node.js 22** with npm for web frontend development
- **Java 17** (Temurin distribution) for Java-based tools
- **Rust** toolchain for building Krakatau
- **Python 3.11** with virtual environment

### EDA Tools
- **ngspice 44.2** - SPICE circuit simulator (compiled from source)
- **OpenVAF 23.5.0** - Verilog-A compiler
- **Krakatau (krak2)** - Java bytecode analysis toolkit

### Process Design Kits (PDKs)
- **IHP SG13G2** - IHP's 130nm technology PDK
- **SkyWater Sky130A/Sky130B** - Google/SkyWater 130nm technology PDKs

### Development Environment
- VS Code extensions for Python, JavaScript, and other languages
- Port forwarding for the ORDeC server (port 8100)
- Proper environment variables for PDK locations

## Usage

### In GitHub Codespaces
1. Navigate to your fork of the repository
2. Click "Code" → "Codespaces" → "Create codespace on [branch]"
3. Wait for the setup to complete (this may take 10-15 minutes for first-time setup)

### Local Development with VS Code
1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the repository in VS Code
3. When prompted, click "Reopen in Container"

## Setup Process

The container setup process runs automatically and includes:

1. **System packages**: Build tools, compilers, and libraries
2. **ngspice**: Downloads and compiles from source with optimized configuration
3. **OpenVAF**: Downloads and extracts the precompiled binary
4. **PDKs**: Downloads and configures IHP and SkyWater PDKs
5. **Verilog-A compilation**: Compiles models with OpenVAF
6. **Python environment**: Creates virtual environment and installs dependencies
7. **ORDeC installation**: Installs the project in development mode
8. **Web dependencies**: Installs npm packages for the frontend

## Environment Variables

The following environment variables are automatically set:

- `ORDEC_PDK_SKY130A`: Path to SkyWater Sky130A PDK
- `ORDEC_PDK_SKY130B`: Path to SkyWater Sky130B PDK  
- `ORDEC_PDK_IHP_SG13G2`: Path to IHP SG13G2 PDK
- `PATH`: Includes ngspice, openvaf, and local bin directories

## Working with the Environment

### Python Development
```bash
# The virtual environment is automatically activated
python -m pytest  # Run tests
ordec-server      # Start the ORDeC server
```

### Web Development
```bash
cd web
npm run dev       # Start development server
npm run build     # Build for production
```

### Using ngspice
```bash
ngspice           # Start interactive mode
ngspice -b circuit.cir  # Batch mode
```

### Using Krakatau
```bash
krak2 dis MyClass.class     # Disassemble Java class
krak2 asm MyClass.j         # Assemble from bytecode
```

## Troubleshooting

### Setup fails
If the initial setup fails, you can manually run the setup script:
```bash
bash .devcontainer/setup.sh
```

### Missing tools
Check that all tools are in PATH:
```bash
which ngspice openvaf krak2 python node npm
```

### PDK issues
Verify PDK environment variables:
```bash
echo $ORDEC_PDK_SKY130A
echo $ORDEC_PDK_IHP_SG13G2
ls -la /home/vscode/skywater/
ls -la /home/vscode/IHP-Open-PDK/
```

## Files

- `devcontainer.json`: Main configuration file
- `setup.sh`: Setup script that replicates Docker environment