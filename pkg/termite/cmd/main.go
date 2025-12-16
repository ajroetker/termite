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

// Command termite runs the Termite ML inference service.
//
// Termite provides embeddings, chunking, and reranking capabilities using ONNX models.
// It can run as a standalone service or be embedded in Antfly.
//
// Usage:
//
//	termite run                    # Start the server
//	termite pull <model>           # Download a model from the registry
//	termite list                   # List local models
//	termite list --remote          # List available models in registry
package main

import (
	"runtime"

	"github.com/antflydb/termite/pkg/termite/cmd/cmd"
)

// https://goreleaser.com/cookbooks/using-main.version/
//
// By default, GoReleaser will set the following 3 ldflags:
//
// main.version: Current Git tag (the v prefix is stripped) or the name of the snapshot, if you're using the --snapshot flag
var version = "dev"

// main.commit: Current git commit SHA
// commit = "none"
// main.date: Date in the RFC3339 format
// date = "unknown"

func main() {
	runtime.SetMutexProfileFraction(1) // Enable mutex profiling
	runtime.SetBlockProfileRate(1)     // Sample every blocking event
	cmd.Version = version
	cmd.Execute()
}
