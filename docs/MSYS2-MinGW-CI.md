# MSYS2/MinGW CI Workflow for ORDeC

This document describes the MSYS2/MinGW-based CI workflow for building and testing ORDeC on Windows.

## Overview

The `msys2-mingw.yaml` workflow provides a Windows-based build alternative to the existing Docker-based Linux CI. It uses the MSYS2 environment with MinGW-w64 to build native Windows binaries of ORDeC and its dependencies.

## Workflow Structure

### Environment Setup
- **Runner**: `windows-latest`
- **Shell**: MSYS2 bash shell
- **Toolchain**: MinGW-w64 x86_64

### Key Dependencies Installed
- **Build Tools**: gcc, gcc-fortran, autotools, make, pkg-config
- **Python Environment**: Python 3.x with scientific packages (numpy, scipy)
- **Node.js**: For web component building
- **Development Tools**: git, wget, tar, unzip, zstd

### Build Steps

1. **Environment Setup**
   - Checkout source code
   - Install MSYS2 and required packages
   - Set up environment variables

2. **Ngspice Build**
   - Download ngspice 44.2 source
   - Configure with MinGW-specific flags
   - Build both minimal and shared library versions
   - Install to local directories

3. **External Dependencies**
   - Download OpenVAF (Linux binary as placeholder - needs Windows version)
   - Clone IHP-Open-PDK repository
   - Download Skywater PDK archives

4. **Python Environment**
   - Create virtual environment
   - Install ORDeC dependencies
   - Handle Windows-specific package issues

5. **Web Components**
   - Install npm dependencies
   - Build web assets

6. **Testing**
   - Install ORDeC in development mode
   - Run pytest with coverage
   - Generate test reports

## Key Differences from Linux Build

### Library Paths
- Uses `PATH` instead of `LD_LIBRARY_PATH` for DLL loading
- Windows-style path separators and library extensions

### Virtual Environment
- Uses `venv/Scripts/activate` instead of `venv/bin/activate`
- Different path structure for Python executables

### Package Availability
- Some packages (like `inotify-simple`) may not work on Windows
- Graceful handling of missing packages with warnings

### OpenVAF Compatibility
- Currently downloads Linux binary as placeholder
- Needs actual Windows-compatible OpenVAF binary
- May require Wine or alternative approach

## Known Limitations

1. **OpenVAF**: No native Windows binary available, using Linux version as placeholder
2. **inotify-simple**: Linux-specific package, may cause functionality limitations
3. **Path Handling**: Some code paths may need Windows-specific adjustments
4. **Performance**: MSYS2 build may be slower than native Linux builds

## Triggering the Workflow

The workflow can be triggered by:
- Manual dispatch (`workflow_dispatch`)
- Push to `main` branch
- Pull requests to `main` branch

## Artifacts

The workflow produces:
- Test results (JUnit XML format)
- Coverage reports (HTML)
- Built wheel packages
- Compiled binaries (ngspice, openvaf)

## Future Improvements

1. **Native Windows OpenVAF**: Obtain or build Windows-compatible OpenVAF binary
2. **Windows-specific Testing**: Add Windows-specific test cases
3. **Performance Optimization**: Optimize build times and caching
4. **Cross-compilation**: Consider building Windows binaries on Linux
5. **Package Distribution**: Windows installer or package generation

## Usage

To use this workflow:

1. Ensure your repository has the workflow file: `.github/workflows/msys2-mingw.yaml`
2. Push changes or create a pull request
3. Monitor the workflow execution in GitHub Actions
4. Download artifacts from successful runs

## Troubleshooting

Common issues and solutions:

- **Build Failures**: Check MSYS2 package availability and MinGW compatibility
- **Test Failures**: Verify Windows-specific path handling and package availability
- **Performance Issues**: Consider using GitHub Actions caching for dependencies
- **Missing Dependencies**: Update MSYS2 package list or find Windows alternatives