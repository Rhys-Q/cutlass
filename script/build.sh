#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <example_path>"
    echo "Example: $0 examples/00_basic_gemm"
    exit 1
fi

EXAMPLE_PATH="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$PROJECT_ROOT/cutlass_workspace"

# Check if example path exists
if [ ! -d "$PROJECT_ROOT/$EXAMPLE_PATH" ]; then
    echo "Error: $EXAMPLE_PATH does not exist"
    exit 1
fi

# Create workspace directory if it doesn't exist
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Creating workspace directory: $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR"
fi

# Extract example name from path
EXAMPLE_NAME=$(basename "$EXAMPLE_PATH")
TARGET_DIR="$WORKSPACE_DIR/$EXAMPLE_NAME"

# Remove existing target directory if it exists
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing $TARGET_DIR"
    rm -rf "$TARGET_DIR"
fi

# Copy example to workspace
echo "Copying $EXAMPLE_PATH to $TARGET_DIR"
cp -r "$PROJECT_ROOT/$EXAMPLE_PATH" "$TARGET_DIR"
# copy cmake file
cp "$PROJECT_ROOT/cmake/example.cmake" "$TARGET_DIR/CMakeLists.txt"
# Change to target directory and build
cd "$TARGET_DIR"
echo "Building in $TARGET_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCUTLASS_DIR="$PROJECT_ROOT"

# Build
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable located in: $TARGET_DIR/build"
