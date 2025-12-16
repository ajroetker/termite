# Hugot Session Factory

This package provides a unified interface for creating Hugot sessions with build-tag based backend selection.

## Backend Selection

### Pure Go Backend (Default)

By default, the package uses the pure Go backend via goMLX. This requires no CGO and produces a single static binary.

**Build:**
```bash
go build
```

**Advantages:**
- No CGO required
- Single static binary
- Easy cross-compilation
- Works everywhere

**Trade-offs:**
- Slower inference compared to ONNX Runtime
- Best for small to medium batches (~32 inputs)

### ONNX Runtime Backend

For production deployments requiring maximum performance, use the ONNX Runtime backend.

**Build:**
```bash
CGO_ENABLED=1 go build -tags="onnx,ORT"
```

**Requirements:**
1. CGO must be enabled
2. Install libomp: `brew install libomp`
3. ONNX Runtime libraries must be available

**Advantages:**
- Much faster inference (native C++ implementation)
- Hardware acceleration support
- Production-grade performance

## Environment Variables

### ONNX Runtime Backend

When using the ONNX Runtime backend, you must set library paths for the dynamic linker:

#### DYLD_LIBRARY_PATH (macOS) / LD_LIBRARY_PATH (Linux)

**Required at runtime** for the dynamic linker to find ONNX Runtime shared libraries.

**Examples:**

```bash
# macOS with Homebrew installation
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib

# macOS with manual download
export DYLD_LIBRARY_PATH=~/Downloads/onnxruntime-osx-arm64-1.23.2/lib

# Linux
export LD_LIBRARY_PATH=/usr/local/lib

# Then run your application
./myapp

# Or set inline
DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib ./myapp
```

**Note:** These environment variables must be set **before** starting your application. The dynamic linker reads them when the process starts, before any Go code runs.

## Installation

### Installing ONNX Runtime on macOS

#### Option 1: Homebrew (Recommended)

The easiest way to install ONNX Runtime on macOS:

1. **Install ONNX Runtime and libomp:**
   ```bash
   brew install onnxruntime libomp
   ```

2. **Download tokenizers library (required):**
   ```bash
   cd ~/Downloads
   curl -L -o libtokenizers.darwin-arm64.tar.gz \
     https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-arm64.tar.gz
   tar -xzf libtokenizers.darwin-arm64.tar.gz
   ```

3. **Set environment variables:**
   ```bash
   # Runtime: ONNX Runtime library path for dynamic linker
   export DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib

   # Build: Tokenizers library path for linker
   export CGO_LDFLAGS="-L$HOME/Downloads -ltokenizers"
   ```

4. **Build with ONNX support:**
   ```bash
   CGO_ENABLED=1 go build -tags="onnx,ORT"
   ```

#### Option 2: Download from GitHub Releases

Alternatively, download directly from Microsoft:

1. **Install libomp:**
   ```bash
   brew install libomp
   ```

2. **Download ONNX Runtime:**
   ```bash
   curl -L -o onnxruntime-osx-arm64.tgz \
     https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-osx-arm64-1.23.2.tgz
   tar -xzf onnxruntime-osx-arm64.tgz
   ```

3. **Download tokenizers library (required):**
   ```bash
   curl -L -o libtokenizers.darwin-arm64.tar.gz \
     https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-arm64.tar.gz
   tar -xzf libtokenizers.darwin-arm64.tar.gz
   ```

4. **Set environment variables:**
   ```bash
   # Runtime: ONNX Runtime library path for dynamic linker
   export DYLD_LIBRARY_PATH=~/Downloads/onnxruntime-osx-arm64-1.23.2/lib

   # Build: Tokenizers library path for linker
   export CGO_LDFLAGS="-L$HOME/Downloads -ltokenizers"
   ```

5. **Build with ONNX support:**
   ```bash
   CGO_ENABLED=1 go build -tags="onnx,ORT"
   ```

### Installing Tokenizers Library

The tokenizers library (`libtokenizers.a`) is required by Hugot when using ONNX Runtime.

**Download prebuilt library for macOS ARM64:**

```bash
# Download from daulet/tokenizers releases
curl -L -o libtokenizers.darwin-arm64.tar.gz \
  https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-arm64.tar.gz

# Extract the archive
tar -xzf libtokenizers.darwin-arm64.tar.gz

# The libtokenizers.a file is now ready to use
```

**Set the library path:**

You can either:

1. **Move to a standard location:**
   ```bash
   sudo cp libtokenizers.a /usr/local/lib/
   export CGO_LDFLAGS="-L/usr/local/lib -ltokenizers"
   ```

2. **Keep in Downloads and reference it:**
   ```bash
   export CGO_LDFLAGS="-L/Users/yourusername/Downloads -ltokenizers"
   ```

**Alternative sources:**
- [hugot releases](https://github.com/knights-analytics/hugot/releases) (check for tokenizers binaries)
- [daulet/tokenizers releases](https://github.com/daulet/tokenizers/releases) (official source)

## Downloading Models

Antfly uses Hugging Face models for embeddings, reranking, and chunking. You can download them using the Hugging Face CLI.

### Install Hugging Face CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Download Required Models

**Embedder (BGE Small):**
```bash
hf download BAAI/bge-small-en-v1.5
```

**Reranker (MXBai):**
```bash
hf download mixedbread-ai/mxbai-rerank-base-v1
```

**Chunker (Chonky):**
```bash
hf download mirth/chonky_mmbert_small_multilingual_1
```

By default, models are downloaded to `~/.cache/huggingface/hub/`. You can point Antfly to these models using the `HUGOT_MODEL_PATH` environment variable or configure them in your application settings.

## Usage Example

```go
package main

import (
    "log"

    "github.com/antflydb/antfly/termite/lib/hugot"
)

func main() {
    // Create session - backend is selected at build time
    session, err := hugot.NewSession()
    if err != nil {
        log.Fatalf("Failed to create session: %v", err)
    }
    defer session.Destroy()

    log.Printf("Using backend: %s", hugot.BackendName())

    // Use session...
}
```

## Makefile Integration

Add these targets to your Makefile:

```makefile
# Build with pure Go backend (default)
build:
	go build -o antfly ./cmd/antfly

# Build with ONNX Runtime backend
build-onnx:
	@echo "Building with ONNX Runtime backend..."
	CGO_ENABLED=1 go build -tags="onnx,ORT" -o antfly ./cmd/antfly
```

## Benchmarking

To compare backends, run benchmarks:

```bash
# Pure Go backend (goMLX)
HUGOT_MODEL_PATH=$(pwd)/models/embedders/bge_small_onnx \
  go test -bench=BenchmarkHugotEmbedder_ShortTexts -benchmem -run='^$' ./lib/embeddings/

# ONNX Runtime backend (requires libraries installed)
DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib \
  ONNX_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib \
  CGO_ENABLED=1 \
  CGO_LDFLAGS="-L/Users/yourusername/Downloads -ltokenizers" \
  HUGOT_MODEL_PATH=$(pwd)/models/embedders/bge_small_onnx \
  go test -tags="onnx,ORT" -bench=BenchmarkHugotEmbedder_ShortTexts -benchmem -run='^$' ./lib/embeddings/
```

### Performance Results

On Apple M4 Max with bge-small model and 10 short texts:

**Pure Go Backend (goMLX):**
- Time per operation: ~137ms
- Memory per operation: ~28.4 MB
- Allocations: ~57,942

**ONNX Runtime Backend:**
- Time per operation: ~8.6ms ⚡️ **16x faster**
- Memory per operation: ~0.32 MB 💾 **89x less memory**
- Allocations: ~318 🔧 **182x fewer allocations**

## Troubleshooting

### Library not found errors

If you see errors like `library 'tokenizers' not found` at **build time**:
1. Ensure `CGO_LDFLAGS` points to libtokenizers.a:
   ```bash
   export CGO_LDFLAGS="-L$HOME/Downloads -ltokenizers"
   ls -la $HOME/Downloads/libtokenizers.a
   ```

If you see errors like `library 'onnxruntime' not found` at **runtime**:
1. Ensure `DYLD_LIBRARY_PATH` is set before running:
   ```bash
   export DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib
   ls -la $DYLD_LIBRARY_PATH/libonnxruntime*.dylib
   ```
2. Verify libomp is installed:
   ```bash
   brew list libomp
   ```

### Code signing issues

On macOS, you may need to allow unsigned libraries:
```bash
sudo xattr -r -d com.apple.quarantine ~/Downloads/onnxruntime-osx-arm64-1.23.2
```

### Build tag combinations

Always use both tags together for ONNX Runtime:
```bash
# Correct
go build -tags="onnx,ORT"

# Wrong - will use pure Go backend
go build -tags="onnx"
```
