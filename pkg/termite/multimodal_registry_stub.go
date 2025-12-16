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

//go:build !(onnx && ORT)

package termite

import (
	"fmt"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	"go.uber.org/zap"
)

// MultimodalEmbedderRegistry is a stub when built without ONNX support.
// To enable CLIP multimodal embeddings, build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type MultimodalEmbedderRegistry struct {
	logger *zap.Logger
}

// NewMultimodalEmbedderRegistry creates a stub registry when CLIP support is disabled.
func NewMultimodalEmbedderRegistry(modelsDir string, logger *zap.Logger) (*MultimodalEmbedderRegistry, error) {
	if modelsDir != "" {
		logger.Warn("Multimodal embeddings (CLIP) not available - build with -tags=\"onnx,ORT\" to enable",
			zap.String("dir", modelsDir))
	}
	return &MultimodalEmbedderRegistry{logger: logger}, nil
}

// Get always returns an error when CLIP support is disabled.
func (r *MultimodalEmbedderRegistry) Get(modelName string) (embeddings.Embedder, error) {
	return nil, fmt.Errorf("multimodal model %s not available: build with -tags=\"onnx,ORT\" to enable CLIP support", modelName)
}

// List returns an empty list when CLIP support is disabled.
func (r *MultimodalEmbedderRegistry) List() []string {
	return nil
}

// Close is a no-op when CLIP support is disabled.
func (r *MultimodalEmbedderRegistry) Close() error {
	return nil
}
