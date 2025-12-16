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

package embeddings

import (
	"errors"

	"go.uber.org/zap"
)

// CLIPEmbedder is a stub when built without ONNX support.
// To enable CLIP multimodal embeddings, build with: CGO_ENABLED=1 go build -tags="onnx,ORT"
type CLIPEmbedder struct{}

// NewCLIPEmbedder returns an error when CLIP support is disabled.
func NewCLIPEmbedder(modelPath string, quantized bool, logger *zap.Logger) (*CLIPEmbedder, error) {
	return nil, errors.New("CLIP embedder not available: build with -tags=\"onnx,ORT\" to enable")
}
