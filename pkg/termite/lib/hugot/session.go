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

// Package hugot provides a unified interface for creating Hugot sessions
// with build-tag based backend selection.
//
// By default (no build tags), this uses the pure Go backend via goMLX.
// With the 'onnx' build tag, it uses ONNX Runtime for faster inference (requires CGO).
//
// Example usage:
//
//	session, err := hugot.NewSession()  // Uses appropriate backend based on build tags
package hugot

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// NewSession creates a new Hugot session using the backend selected at build time.
//
// Without build tags (default): Uses pure Go backend (goMLX) - no CGO required
// With -tags=onnx: Uses ONNX Runtime backend - requires CGO, faster inference
//
// The implementation of this function is provided by either:
//   - session_go.go (default, pure Go backend)
//   - session_onnx.go (ONNX Runtime backend, requires -tags=onnx)
func NewSession(opts ...options.WithOption) (*hugot.Session, error) {
	return newSessionImpl(opts...)
}

// NewSessionOrUseExisting returns the provided session if non-nil, otherwise creates a new one.
// This is useful when you want to share a single session across multiple models/pipelines.
//
// IMPORTANT: With ONNX Runtime backend, only ONE session can be active at a time.
// Use this function to share a session across chunkers, rerankers, and embedders.
//
// Example:
//
//	// Create shared session once
//	sharedSession, err := hugot.NewSession()
//	if err != nil {
//		return err
//	}
//	defer sharedSession.Destroy()
//
//	// Reuse session for multiple models
//	session1, _ := hugot.NewSessionOrUseExisting(sharedSession)  // Returns sharedSession
//	session2, _ := hugot.NewSessionOrUseExisting(sharedSession)  // Returns sharedSession
//	session3, _ := hugot.NewSessionOrUseExisting(nil)            // Creates new session
func NewSessionOrUseExisting(existingSession *hugot.Session, opts ...options.WithOption) (*hugot.Session, error) {
	if existingSession != nil {
		return existingSession, nil
	}
	return newSessionImpl(opts...)
}

// BackendName returns a human-readable name of the backend being used.
// Useful for logging and debugging.
func BackendName() string {
	return backendNameImpl()
}

// GetGPUInfo returns information about detected GPU hardware.
// This is always available regardless of build tags.
func GetGPUInfo() GPUInfo {
	return DetectGPU()
}
