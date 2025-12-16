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

//go:build onnx && ORT

package termite

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/antflydb/antfly-go/libaf/embeddings"
	termembeddings "github.com/antflydb/termite/pkg/termite/lib/embeddings"
	"go.uber.org/zap"
)

// MultimodalEmbedderRegistry manages CLIP and other multimodal embedding models.
// These models have separate visual and text encoders and can embed both images and text
// into a shared embedding space.
//
// Build with -tags="onnx,ORT" to enable this registry.
type MultimodalEmbedderRegistry struct {
	models map[string]embeddings.Embedder
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewMultimodalEmbedderRegistry creates a registry for multimodal models.
// It scans the models directory for CLIP-style models containing:
//   - visual_model.onnx (or visual_model_quantized.onnx)
//   - text_model.onnx (or text_model_quantized.onnx)
func NewMultimodalEmbedderRegistry(modelsDir string, logger *zap.Logger) (*MultimodalEmbedderRegistry, error) {
	registry := &MultimodalEmbedderRegistry{
		models: make(map[string]embeddings.Embedder),
		logger: logger,
	}

	if modelsDir == "" {
		logger.Info("No multimodal models directory configured")
		return registry, nil
	}

	// Check if directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		logger.Warn("Multimodal models directory does not exist",
			zap.String("dir", modelsDir))
		return registry, nil
	}

	// Scan directory for model subdirectories
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("reading models directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelName := entry.Name()
		modelPath := filepath.Join(modelsDir, modelName)

		// Check for CLIP-style model structure
		visualPath := filepath.Join(modelPath, "visual_model.onnx")
		textPath := filepath.Join(modelPath, "text_model.onnx")
		visualQuantizedPath := filepath.Join(modelPath, "visual_model_quantized.onnx")
		textQuantizedPath := filepath.Join(modelPath, "text_model_quantized.onnx")

		hasStandard := fileExists(visualPath) && fileExists(textPath)
		hasQuantized := fileExists(visualQuantizedPath) && fileExists(textQuantizedPath)

		if !hasStandard && !hasQuantized {
			logger.Debug("Skipping directory without CLIP model files",
				zap.String("dir", modelName))
			continue
		}

		logger.Info("Discovered multimodal model directory",
			zap.String("name", modelName),
			zap.String("path", modelPath),
			zap.Bool("has_standard", hasStandard),
			zap.Bool("has_quantized", hasQuantized))

		// Load standard precision model if it exists
		if hasStandard {
			model, err := termembeddings.NewCLIPEmbedder(modelPath, false, logger.Named(modelName))
			if err != nil {
				logger.Warn("Failed to load standard CLIP model",
					zap.String("name", modelName),
					zap.Error(err))
			} else {
				registry.models[modelName] = model
				logger.Info("Successfully loaded standard CLIP model",
					zap.String("name", modelName))
			}
		}

		// Load quantized model if it exists (register with -i8-qt suffix)
		if hasQuantized {
			quantizedName := modelName + "-i8-qt"
			model, err := termembeddings.NewCLIPEmbedder(modelPath, true, logger.Named(quantizedName))
			if err != nil {
				logger.Warn("Failed to load quantized CLIP model",
					zap.String("name", quantizedName),
					zap.Error(err))
			} else {
				registry.models[quantizedName] = model
				logger.Info("Successfully loaded quantized CLIP model",
					zap.String("name", quantizedName))
			}
		}
	}

	logger.Info("Multimodal embedder registry initialized",
		zap.Int("models_loaded", len(registry.models)))

	return registry, nil
}

// Get returns an embedder by model name
func (r *MultimodalEmbedderRegistry) Get(modelName string) (embeddings.Embedder, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[modelName]
	if !ok {
		return nil, fmt.Errorf("multimodal model not found: %s", modelName)
	}
	return model, nil
}

// List returns all available model names
func (r *MultimodalEmbedderRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.models))
	for name := range r.models {
		names = append(names, name)
	}
	return names
}

// Close closes all loaded models
func (r *MultimodalEmbedderRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	for name, model := range r.models {
		if clipEmb, ok := model.(*termembeddings.CLIPEmbedder); ok {
			if err := clipEmb.Close(); err != nil {
				r.logger.Warn("Error closing CLIP model",
					zap.String("name", name),
					zap.Error(err))
			}
		}
	}
	return nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
