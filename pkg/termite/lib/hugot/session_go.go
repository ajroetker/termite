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

//go:build !(onnx && ORT) && !(xla && XLA)

package hugot

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// newSessionImpl creates a Hugot session using the pure Go backend (goMLX).
// This is the default implementation when no build tags are specified.
//
// Advantages:
//   - No CGO required
//   - Single static binary
//   - Easy cross-compilation
//   - Works everywhere
//
// Trade-offs:
//   - Slower inference compared to ONNX Runtime
func newSessionImpl(opts ...options.WithOption) (*hugot.Session, error) {
	return hugot.NewGoSession(opts...)
}

// backendNameImpl returns the name of the pure Go backend.
func backendNameImpl() string {
	return "goMLX (Pure Go)"
}

// SetGPUMode is a no-op for the pure Go backend.
// GPU acceleration requires the ONNX Runtime backend (build with -tags onnx,ORT).
func SetGPUMode(mode GPUMode) {
	// No-op: pure Go backend doesn't support GPU acceleration
}

// GetGPUMode returns GPUModeOff for the pure Go backend.
// GPU acceleration requires the ONNX Runtime backend.
func GetGPUMode() GPUMode {
	return GPUModeOff
}
