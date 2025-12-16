// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build onnx && ORT && darwin

package hugot

import (
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

var (
	// configuredGPUMode is set via SetGPUMode or config
	configuredGPUMode GPUMode = GPUModeAuto
	gpuModeMu         sync.RWMutex
)

// newSessionImpl creates a Hugot session using ONNX Runtime with CoreML acceleration.
// This implementation is used by default on macOS (darwin) with -tags="onnx,ORT".
//
// CoreML provides hardware acceleration on Apple Silicon and Intel Macs with
// Neural Engine, GPU, or CPU execution depending on the model and hardware.
//
// Runtime Requirements:
//   - Set DYLD_LIBRARY_PATH before running:
//     export DYLD_LIBRARY_PATH=/opt/homebrew/opt/onnxruntime/lib
//
// Build Requirements:
//   - CGO must be enabled (CGO_ENABLED=1)
//   - ONNX Runtime libraries must be available at link time
//   - libomp installed (brew install libomp)
//   - Tokenizers library available (CGO_LDFLAGS)
func newSessionImpl(opts ...options.WithOption) (*hugot.Session, error) {
	// Prepend CoreML provider - user options can override if needed
	coremlOpts := []options.WithOption{options.WithCoreML(nil)}
	opts = append(coremlOpts, opts...)
	return hugot.NewORTSession(opts...)
}

// backendNameImpl returns the name of the ONNX Runtime backend with CoreML.
func backendNameImpl() string {
	return "ONNX Runtime (CoreML)"
}

// SetGPUMode sets the GPU mode for future sessions.
// On macOS, CoreML automatically uses the best available accelerator
// (Neural Engine, GPU, or CPU), so this is mostly informational.
func SetGPUMode(mode GPUMode) {
	gpuModeMu.Lock()
	defer gpuModeMu.Unlock()
	configuredGPUMode = mode
}

// GetGPUMode returns the currently configured GPU mode.
func GetGPUMode() GPUMode {
	gpuModeMu.RLock()
	defer gpuModeMu.RUnlock()
	return configuredGPUMode
}
